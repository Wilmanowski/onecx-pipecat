import warnings

import pytest
from pydantic import SecretStr

from onecx_pipecat.config import CredentialsSettings, Settings
from onecx_pipecat.models import AgentConfigSnapshot, ModelSnapshot, ProviderSnapshot
from onecx_pipecat.services.credentials import CredentialError, CredentialResolver
from onecx_pipecat.services.llm import LlmConfigError, build_llm, parse_model_params

_REF = "credential://provider/p1/api-key"


def _snap(ptype="OPENAI", *, url="", auth="API_KEY", ref=_REF, params="", ident="m-1"):
    return AgentConfigSnapshot(
        agent_id="a",
        schema_version="1",
        provider=ProviderSnapshot(type=ptype, llm_url=url, auth_mode=auth, credential_ref=ref),
        model=ModelSnapshot(model_identifier=ident, model_params=params),
    )


def _build(snapshot):
    creds = CredentialResolver(CredentialsSettings(refs={_REF: SecretStr("sk-test")}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return build_llm(snapshot, Settings(), creds)


def test_openai():
    from pipecat.services.openai.llm import OpenAILLMService

    llm = _build(
        _snap(url="https://llm.example/v1", params='{"temperature": 0.3, "maxTokens": 99}')
    )
    assert isinstance(llm, OpenAILLMService)
    assert llm._settings.model == "m-1"
    assert llm._settings.temperature == 0.3
    assert llm._settings.max_tokens == 99


def test_openai_without_key_needs_base_url():
    with pytest.raises(LlmConfigError):
        _build(_snap(auth="", ref=""))
    assert _build(_snap(auth="", ref="", url="http://local:1234/v1")) is not None


def test_ollama():
    from pipecat.services.ollama.llm import OLLamaLLMService

    llm = _build(_snap("OLLAMA", url="http://ollama:11434/v1", auth="", ref=""))
    assert isinstance(llm, OLLamaLLMService)
    assert llm._settings.model == "m-1"


def test_anthropic():
    from pipecat.services.anthropic.llm import AnthropicLLMService

    llm = _build(_snap("ANTHROPIC", params='{"top_p": 0.5}'))
    assert isinstance(llm, AnthropicLLMService)
    assert llm._settings.top_p == 0.5


def test_anthropic_with_custom_url_and_without_key():
    assert _build(_snap("ANTHROPIC", url="https://proxy.example")) is not None
    with pytest.raises(LlmConfigError):
        _build(_snap("ANTHROPIC", auth="", ref=""))


def test_unknown_credential_ref_fails_closed():
    with pytest.raises(CredentialError):
        _build(_snap(ref="credential://other"))


@pytest.mark.parametrize(
    "snapshot",
    [
        _snap("BEDROCK"),
        _snap(auth="OAUTH"),
        _snap(params="{not json"),
        _snap(params="[1]"),
        _snap(params='{"temperature": "hot"}'),
        AgentConfigSnapshot(agent_id="a", schema_version="1"),
    ],
)
def test_invalid_config_raises(snapshot):
    with pytest.raises(LlmConfigError):
        _build(snapshot)


def test_parse_model_params_ignores_unknown_keys():
    assert parse_model_params('{"temperature": 1, "foo": "bar", "max_tokens": 5.0}') == {
        "temperature": 1.0,
        "max_tokens": 5,
    }
    assert parse_model_params("") == {}


@pytest.mark.parametrize(
    ("raw", "api", "effort"),
    [
        ("", "chat-completions", None),
        ("{}", "chat-completions", None),
        ('{"api":""}', "chat-completions", None),
        ('{"api":"  "}', "chat-completions", None),
        ('{"api":"responses","reasoningEffort":" medium "}', "responses", "medium"),
        ('{"api":"chat-completions","reasoningEffort":"  "}', "chat-completions", None),
        (
            '{"reasoningEffort":" future-effort ","unknown":true}',
            "chat-completions",
            "future-effort",
        ),
    ],
)
def test_parse_openai_options(raw, api, effort):
    from onecx_pipecat.services.llm import parse_openai_model_options

    options = parse_openai_model_options(raw)
    assert options.api == api
    assert options.reasoning_effort == effort


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        "{} {}",
        "[]",
        '{"api":"legacy"}',
        '{"api":" responses "}',
        '{"api":"Responses"}',
        '{"api":null}',
        '{"api":false}',
        '{"api":1}',
        '{"api":[]}',
        '{"api":{}}',
        '{"reasoningEffort":null}',
        '{"reasoningEffort":false}',
        '{"reasoningEffort":1}',
        '{"reasoningEffort":[]}',
        '{"reasoningEffort":{}}',
        '{"api":"responses","api":"chat-completions"}',
        '{"temperature":1,"temperature":2}',
        '{"unknown":{"key":1,"key":2}}',
        '{"unknown":"' + "x" * 242 + '"}',
        " " * 256,
    ],
)
def test_parse_openai_options_rejects_invalid_config(raw):
    from onecx_pipecat.services.llm import parse_openai_model_options

    with pytest.raises(LlmConfigError):
        parse_openai_model_options(raw)
    with pytest.raises(LlmConfigError):
        _build(_snap(params=raw))


def test_openai_options_preserve_numeric_params_at_size_limit():
    from onecx_pipecat.services.llm import parse_openai_model_options

    raw = '{"temperature":0.3,"maxTokens":99,"top_p":0.5,"api":"responses"}'.ljust(255)
    assert parse_openai_model_options(raw).api == "responses"
    assert parse_model_params(raw) == {"temperature": 0.3, "max_tokens": 99, "top_p": 0.5}


@pytest.mark.parametrize("api", ["", "chat-completions"])
@pytest.mark.parametrize("effort", ["", "   ", " future-effort ", " none "])
def test_chat_completions_reasoning_settings(api, effort):
    import json

    from pipecat.services.openai.llm import OpenAILLMService

    llm = _build(_snap(params=json.dumps({"api": api, "reasoningEffort": effort})))
    assert type(llm) is OpenAILLMService
    assert llm._settings.extra == ({"reasoning_effort": effort.strip()} if effort.strip() else {})


def test_responses_service_preserves_credentials_url_and_numeric_settings():
    from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService

    llm = _build(
        _snap(
            url="https://llm.example/v1",
            ident="provider-model",
            params=(
                '{"api":"responses","reasoningEffort":" medium ",'
                '"temperature":0.3,"maxTokens":99,"top_p":0.5,"unknown":"ignored"}'
            ),
        )
    )
    assert isinstance(llm, OpenAIResponsesHttpLLMService)
    assert llm._settings.model == "provider-model"
    assert llm._client.api_key == "sk-test"
    assert str(llm._client.base_url) == "https://llm.example/v1/"
    assert llm._settings.reasoning.effort == "medium"
    assert llm._settings.temperature == 0.3
    assert llm._settings.max_completion_tokens == 99
    assert llm._settings.top_p == 0.5
    invocation = {"input": [{"role": "user", "content": "Hi"}], "tools": []}
    params = llm._build_response_params(invocation)
    assert params == {
        "model": "provider-model",
        "stream": True,
        "store": False,
        "input": [{"role": "user", "content": "Hi"}],
        "tools": [],
        "temperature": 0.3,
        "top_p": 0.5,
        "max_output_tokens": 99,
        "reasoning": {"effort": "medium"},
        "include": ["reasoning.encrypted_content"],
    }


@pytest.mark.parametrize(
    ("config", "reasoning"),
    [
        ('{"api":"responses"}', None),
        ('{"api":"responses","reasoningEffort":"  "}', None),
        ('{"api":"responses","reasoningEffort":" none "}', {"effort": "none"}),
        ('{"api":"responses","reasoningEffort":" future-effort "}', {"effort": "future-effort"}),
    ],
)
@pytest.mark.parametrize("model", ["gpt-5.4", "o3", "provider-model"])
def test_responses_request_reasoning_respects_provider_defaults(config, reasoning, model):
    from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService

    llm = _build(_snap(params=config, ident=model, url="http://local/v1", auth="", ref=""))
    assert isinstance(llm, OpenAIResponsesHttpLLMService)
    assert llm._client.api_key == "not-needed"
    params = llm._build_response_params({"input": []})
    if reasoning is None:
        assert llm._settings.reasoning is None
        assert "reasoning" not in params
    else:
        assert params["reasoning"] == reasoning
    if reasoning is None or reasoning["effort"] == "none":
        assert "include" not in params
    else:
        assert params["include"] == ["reasoning.encrypted_content"]
    assert params["store"] is False


@pytest.mark.parametrize("ptype", ["OLLAMA", "ANTHROPIC"])
def test_openai_validation_does_not_change_other_providers(ptype):
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.services.ollama.llm import OLLamaLLMService

    raw = (
        '{"api":[],"reasoningEffort":false,"unknown":1,"unknown":2,'
        '"temperature":0.3,"max_tokens":99,"topP":0.5}'
    ).ljust(256)
    llm = _build(_snap(ptype, params=raw))
    assert isinstance(llm, OLLamaLLMService if ptype == "OLLAMA" else AnthropicLLMService)
    assert llm._settings.temperature == 0.3
    assert llm._settings.max_tokens == 99
    assert llm._settings.top_p == 0.5


@pytest.mark.parametrize("key", ["maxTokens", "max_tokens"])
def test_responses_accepts_existing_token_aliases(key):
    from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService

    llm = _build(_snap(params='{"api":"responses","' + key + '":99}'))
    assert isinstance(llm, OpenAIResponsesHttpLLMService)
    assert llm._build_response_params({"input": []})["max_output_tokens"] == 99


@pytest.mark.parametrize(
    "numeric",
    ['"temperature":"hot"', '"top_p":true', '"maxTokens":"many"'],
)
def test_responses_rejects_invalid_numeric_settings(numeric):
    with pytest.raises(LlmConfigError):
        _build(_snap(params='{"api":"responses",' + numeric + "}"))
