"""Build the Pipecat LLM service for the pinned agent's provider and model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from onecx_pipecat.config import Settings
from onecx_pipecat.models import AgentConfigSnapshot
from onecx_pipecat.services.credentials import CredentialResolver

if TYPE_CHECKING:
    from pipecat.services.llm_service import LLMService

# model_config JSON key (camelCase or snake_case) -> Pipecat LLM settings field.
_PARAM_KEYS: dict[str, str] = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "maxTokens": "max_tokens",
    "top_p": "top_p",
    "topP": "top_p",
}
_INT_PARAMS = {"max_tokens"}

# Placeholder key for OpenAI-compatible endpoints that need none; the client requires a value.
_NO_API_KEY = "not-needed"


class LlmConfigError(Exception):
    """Raised when the pinned provider/model cannot be turned into an LLM service."""


@dataclass(frozen=True)
class OpenAIModelOptions:
    """OpenAI-specific transport and provider-dependent reasoning settings."""

    api: Literal["chat-completions", "responses"] = "chat-completions"
    reasoning_effort: str | None = None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise LlmConfigError(f"model configuration has duplicate key {key!r}")
        data[key] = value
    return data


def parse_openai_model_options(raw: str) -> OpenAIModelOptions:
    """Parse OpenAI options, enforcing the runtime's 255-character config limit."""
    if len(raw) > 255:
        raise LlmConfigError("OpenAI model configuration exceeds 255 characters")
    if not raw.strip():
        return OpenAIModelOptions()
    try:
        data = json.loads(raw, object_pairs_hook=_unique_json_object)
    except ValueError as exc:
        raise LlmConfigError("model configuration is not valid JSON") from exc
    if not isinstance(data, dict):
        raise LlmConfigError("model configuration must be a JSON object")
    api = data.get("api", "chat-completions")
    if not isinstance(api, str):
        raise LlmConfigError("model configuration 'api' must be a string")
    if not api.strip():
        api = "chat-completions"
    if api not in ("chat-completions", "responses"):
        raise LlmConfigError(f"unsupported OpenAI API: {api!r}")
    effort = data.get("reasoningEffort", "")
    if not isinstance(effort, str):
        raise LlmConfigError("model configuration 'reasoningEffort' must be a string")
    return OpenAIModelOptions(api=api, reasoning_effort=effort.strip() or None)


def parse_model_params(raw: str) -> dict[str, Any]:
    """Map the opaque ``modelConfig`` JSON to Pipecat settings; unknown keys are ignored."""
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise LlmConfigError("model configuration is not valid JSON") from exc
    if not isinstance(data, dict):
        raise LlmConfigError("model configuration must be a JSON object")
    params: dict[str, Any] = {}
    for key, field in _PARAM_KEYS.items():
        if key not in data or data[key] is None:
            continue
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise LlmConfigError(f"model configuration {key!r} must be a number")
        params[field] = int(value) if field in _INT_PARAMS else float(value)
    return params


def build_llm(
    snapshot: AgentConfigSnapshot,
    settings: Settings,
    credentials: CredentialResolver,
) -> LLMService:
    """Create the LLM service for ``snapshot``.

    Raises ``LlmConfigError`` for an unusable provider/model/model_config and
    ``CredentialError`` when the provider's credentialRef cannot be resolved.
    Pipecat service modules are imported lazily so unused providers need no SDK.
    """
    provider, model = snapshot.provider, snapshot.model
    if settings.llm.provider_override == "inmemory":
        from onecx_pipecat.services.inmemory import InMemoryLLM

        return InMemoryLLM()
    if provider is None or model is None:
        raise LlmConfigError("agent configuration has no provider or model")
    params = parse_model_params(model.model_params)
    api_key = _resolve_api_key(provider.auth_mode, provider.credential_ref, credentials)
    ptype = provider.type.upper()

    if ptype == "OPENAI":
        from pipecat.services.openai.llm import OpenAILLMService

        options = parse_openai_model_options(model.model_params)
        if api_key is None and not provider.llm_url:
            raise LlmConfigError("OPENAI provider requires an API key credential")
        kwargs: dict[str, Any] = {"api_key": api_key or _NO_API_KEY}
        if provider.llm_url:
            kwargs["base_url"] = provider.llm_url
        if options.api == "responses":
            from pipecat.services.openai.responses.llm import (
                OpenAIResponsesHttpLLMService,
                OpenAIResponsesReasoningConfig,
            )

            class _ProviderDefaultResponsesHttpLLMService(OpenAIResponsesHttpLLMService):
                def _build_response_params(self, invocation_params: dict[str, Any]) -> dict:
                    request = super()._build_response_params(invocation_params)
                    reasoning = self._settings.reasoning
                    # Pipecat 1.8.1 disables reasoning when unset; retain provider defaults.
                    if reasoning is None or not reasoning.model_dump(exclude_none=True):
                        request.pop("reasoning", None)
                    elif reasoning.effort == "none":
                        request.pop("include", None)
                    return request

            if "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens")
            if options.reasoning_effort is not None:
                params["reasoning"] = OpenAIResponsesReasoningConfig(
                    effort=options.reasoning_effort
                )
            return _ProviderDefaultResponsesHttpLLMService(
                settings=OpenAIResponsesHttpLLMService.Settings(
                    model=model.model_identifier, **params
                ),
                **kwargs,
            )
        if options.reasoning_effort is not None:
            params["extra"] = {"reasoning_effort": options.reasoning_effort}
        return OpenAILLMService(
            settings=OpenAILLMService.Settings(model=model.model_identifier, **params), **kwargs
        )

    if ptype == "OLLAMA":
        from pipecat.services.ollama.llm import OLLamaLLMService

        # Pipecat's Ollama service sends a fixed placeholder key; a credential is ignored.
        kwargs = {"base_url": provider.llm_url} if provider.llm_url else {}
        return OLLamaLLMService(
            settings=OLLamaLLMService.Settings(model=model.model_identifier, **params), **kwargs
        )

    if ptype == "ANTHROPIC":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        if api_key is None:
            raise LlmConfigError("ANTHROPIC provider requires an API key credential")
        kwargs = {}
        if provider.llm_url:
            from anthropic import AsyncAnthropic

            kwargs["client"] = AsyncAnthropic(api_key=api_key, base_url=provider.llm_url)
        return AnthropicLLMService(
            api_key=api_key,
            settings=AnthropicLLMService.Settings(model=model.model_identifier, **params),
            **kwargs,
        )

    raise LlmConfigError(f"unsupported LLM provider type: {provider.type!r}")


def _resolve_api_key(
    auth_mode: str, credential_ref: str, credentials: CredentialResolver
) -> str | None:
    mode = auth_mode.upper()
    if mode == "OAUTH":
        raise LlmConfigError("OAUTH provider authentication is not supported")
    if mode == "API_KEY" or credential_ref:
        return credentials.resolve(credential_ref)
    return None
