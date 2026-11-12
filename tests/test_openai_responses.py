import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI
from pipecat.frames.frames import (
    CancelFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from onecx_pipecat.adapters.mcp import McpToolResult
from onecx_pipecat.config import CredentialsSettings, Settings
from onecx_pipecat.models import (
    AgentConfigSnapshot,
    AgentToolSnapshot,
    ModelSnapshot,
    ProviderSnapshot,
)
from onecx_pipecat.pipeline.captions import CaptionLedger
from onecx_pipecat.pipeline.persistence import attach_persistence
from onecx_pipecat.pipeline.tools import build_tools_schema
from onecx_pipecat.services.credentials import CredentialResolver
from onecx_pipecat.services.llm import build_llm

_CREDENTIAL_REF = "credential://provider/openai/api-key"
_ENCRYPTED_REASONING = "encrypted-reasoning-from-the-provider"


class _ToolInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def invoke_tool(self, server_name: str, tool_name: str, arguments=None):
        self.calls.append((server_name, tool_name, arguments))
        return McpToolResult(ok=True, value="42")


class _OutputCollector(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.errors: list[str] = []
        self.error_ready = asyncio.Event()

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.text.append(frame.text)
            self.error_ready.set()
        await self.push_frame(frame, direction)


def _sse(event: str, sequence: int, **payload: object) -> bytes:
    data = {"type": event, "sequence_number": sequence, **payload}
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _responses_stream(turn: int) -> bytes:
    if turn == 1:
        return b"".join(
            [
                _sse(
                    "response.output_item.done",
                    0,
                    output_index=0,
                    item={
                        "type": "reasoning",
                        "id": "rs_reasoning_1",
                        "summary": [],
                        "encrypted_content": _ENCRYPTED_REASONING,
                    },
                ),
                _sse(
                    "response.output_item.added",
                    1,
                    output_index=1,
                    item={
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_mcp_1",
                        "name": "crm__lookup",
                        "arguments": "",
                    },
                ),
                _sse(
                    "response.function_call_arguments.done",
                    2,
                    output_index=1,
                    item_id="fc_1",
                    name="crm__lookup",
                    arguments='{"query":"meaning of life"}',
                ),
                _sse(
                    "response.output_item.done",
                    3,
                    output_index=1,
                    item={
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_mcp_1",
                        "name": "crm__lookup",
                        "arguments": '{"query":"meaning of life"}',
                    },
                ),
            ]
        )
    return _sse(
        "response.output_text.delta",
        0,
        output_index=0,
        content_index=0,
        item_id="msg_final",
        delta="The answer is 42.",
        logprobs=[],
    )


async def _run_pipeline(
    llm,
    context: LLMContext,
    *,
    invoker: _ToolInvoker | None = None,
    config: AgentConfigSnapshot | None = None,
):
    if invoker is not None and config is not None:
        context.set_tools(
            build_tools_schema(
                llm,
                invoker,
                config,
                [
                    AgentToolSnapshot(
                        tool_name="lookup",
                        server_name="crm",
                        tool_description="Look up a value.",
                        input_schema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    )
                ],
            )
        )

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )
    user, assistant = aggregators.user(), aggregators.assistant()
    entries = []

    async def save(entry):
        entries.append(entry)
        return f"entry-{len(entries)}"

    async def emit(_event):
        pass

    async def fatal_context_failure(error):
        raise error

    context_in, context_out = attach_persistence(
        user,
        assistant,
        save,
        emit,
        captions=CaptionLedger(),
        context=context,
        fatal_context_failure=fatal_context_failure,
    )
    output = _OutputCollector()
    final_text = []
    final_text_ready = asyncio.Event()

    async def on_assistant_turn_stopped(_aggregator, message):
        if message.content.strip():
            final_text.append(message.content.strip())
            final_text_ready.set()

    assistant.event_handler("on_assistant_turn_stopped")(on_assistant_turn_stopped)
    task = PipelineWorker(
        Pipeline(
            [
                user,
                context_in,
                llm,
                context_out,
                output,
                assistant,
            ]
        ),
        params=PipelineParams(),
        enable_rtvi=False,
        idle_timeout_secs=None,
    )

    @task.event_handler("on_pipeline_error")
    async def on_pipeline_error(_task, frame):
        output.errors.append(str(frame.error))
        output.error_ready.set()

    runner = WorkerRunner()
    await runner.add_workers(task)
    await task.queue_frame(UserStartedSpeakingFrame())
    await task.queue_frame(
        TranscriptionFrame(
            text="What is the answer?",
            user_id="user-1",
            timestamp="2026-11-12T10:00:00Z",
        )
    )
    await task.queue_frame(UserStoppedSpeakingFrame())
    runner_task = asyncio.create_task(runner.run())
    try:
        completion_tasks = [
            asyncio.create_task(final_text_ready.wait()),
            asyncio.create_task(output.error_ready.wait()),
        ]
        done, pending = await asyncio.wait(
            completion_tasks,
            timeout=6,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for pending_task in pending:
            pending_task.cancel()
        if not done:
            raise TimeoutError("Responses pipeline produced neither text nor an error")
        await task.queue_frame(CancelFrame())
        await asyncio.wait_for(runner_task, timeout=6)
    finally:
        if not runner_task.done():
            await task.cancel()
            await asyncio.gather(runner_task, return_exceptions=True)
    return output, entries, final_text


def _build_responses_llm(*, url: str, params: str):
    config = AgentConfigSnapshot(
        agent_id="agent-1",
        schema_version="1",
        system_prompt="Use the provided skill before answering.",
        provider=ProviderSnapshot(
            type="OPENAI",
            llm_url=url,
            auth_mode="API_KEY",
            credential_ref=_CREDENTIAL_REF,
        ),
        model=ModelSnapshot(model_identifier="gpt-luna", model_params=params),
    )
    credentials = CredentialResolver(CredentialsSettings(refs={_CREDENTIAL_REF: "test-key"}))
    return config, build_llm(config, Settings(), credentials)


async def test_responses_pipeline_runs_registered_tool_and_continues_encrypted_reasoning():
    requests: list[tuple[str, dict]] = []
    url = "https://responses.test/v1"
    config, llm = _build_responses_llm(
        url=url,
        params='{"api":"responses","reasoningEffort":"medium"}',
    )
    original_client = llm._client

    async def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path != "/v1/responses":
            return httpx.Response(
                404,
                json={"error": {"message": "Expected a Responses API request"}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_responses_stream(len(requests)),
        )

    llm._client = AsyncOpenAI(
        api_key=original_client.api_key,
        base_url=url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    await original_client.close()

    invoker = _ToolInvoker()
    context = LLMContext(messages=[{"role": "system", "content": config.system_prompt}])
    try:
        output, _entries, final_text = await _run_pipeline(
            llm,
            context,
            invoker=invoker,
            config=config,
        )
    finally:
        await llm._client.close()

    assert [path for path, _body in requests] == ["/v1/responses", "/v1/responses"]
    first, followup = [body for _path, body in requests]
    assert first["model"] == "gpt-luna"
    assert first["reasoning"] == {"effort": "medium"}
    assert first["include"] == ["reasoning.encrypted_content"]
    assert first["store"] is False
    assert "temperature" not in first
    assert "top_p" not in first
    assert any(item.get("role") == "user" for item in first["input"])
    assert any(tool.get("name") == "crm__lookup" for tool in first["tools"])
    assert any(
        item.get("type") == "function_call" and item.get("call_id") == "call_mcp_1"
        for item in followup["input"]
    )
    assert any(
        item.get("type") == "function_call_output"
        and item.get("call_id") == "call_mcp_1"
        and "42" in item.get("output", "")
        for item in followup["input"]
    )
    assert any(
        item.get("type") == "reasoning" and item.get("encrypted_content") == _ENCRYPTED_REASONING
        for item in followup["input"]
    )
    assert invoker.calls == [("crm", "lookup", {"query": "meaning of life"})]
    assert output.text == ["The answer is 42."]
    assert final_text == ["The answer is 42."]


@pytest.mark.parametrize("status_code", [400, 404])
async def test_responses_http_errors_do_not_retry_or_fall_back(status_code):
    requests = []
    url = "https://responses.test/v1"
    config, llm = _build_responses_llm(
        url=url,
        params='{"api":"responses","reasoningEffort":"medium"}',
    )
    original_client = llm._client

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(
            status_code,
            json={"error": {"message": "Responses request rejected"}},
        )

    llm._client = AsyncOpenAI(
        api_key=original_client.api_key,
        base_url=url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    await original_client.close()

    context = LLMContext(messages=[{"role": "system", "content": config.system_prompt}])
    try:
        output, _entries, final_text = await _run_pipeline(llm, context)
    finally:
        await llm._client.close()

    assert requests == ["/v1/responses"]
    assert output.errors
    assert final_text == []
