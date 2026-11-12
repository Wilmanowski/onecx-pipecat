# Agent Additional Prompt Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the agent's configured `additionalPrompt` to Pipecat voice sessions and include it in the LLM system prompt.

**Architecture:** Extend the ai-provider public configuration response with optional `agent.additionalPrompt`, populated from the persisted Agent field. Parse that value into Pipecat's pinned snapshot and compose it after the scaffold system prompt, preserving the existing skills and session `ai_context` blocks.

**Tech Stack:** Java, Quarkus, OpenAPI, Maven; Python 3.11+, Pydantic, pytest, Ruff; GitHub CLI for the two PRs.

## Global Constraints

- Keep the field optional so older or incomplete responses continue to work.
- The field is pinned once per voice session with the existing agent configuration.
- Missing or blank `additionalPrompt` leaves the existing prompt unchanged.
- Trim configured prompt strings and join present blocks with a blank line.
- Prompt order: scaffold `systemPrompt`, agent `additionalPrompt`, available-skills block, session `ai_context`.
- Do not change protocol messages, session `ai_context`, skill handling, or unrelated runtime prompt behavior.
- Before exploring source in a repository with `graphify-out/graph.json`, run `graphify query "<question>"` first. The ai-provider service did not have a graph during design exploration; if still absent, proceed with targeted source inspection.
- Create separate service and Pipecat PRs; submit the service PR first and make the Pipecat PR depend on its response contract.

---

## File Structure

### ai-provider service PR

- `../onecx-ai-provider-svc/src/main/openapi/onecx-ai-v1.yaml` defines the response property under `AgentConfigurationAgent`.
- `../onecx-ai-provider-svc/src/main/java/org/tkit/onecx/ai/provider/rs/external/v1/mappers/AgentConfigurationMapper.java` maps the persisted field to the response DTO.
- `../onecx-ai-provider-svc/src/test/java/org/tkit/onecx/ai/provider/rs/external/v1/mappers/AgentConfigurationMapperTest.java` verifies the mapper output.

### Pipecat PR

- `onecx_pipecat/models.py` stores the optional value in `AgentConfigSnapshot`.
- `onecx_pipecat/adapters/agent_config.py` reads `agent.additionalPrompt`.
- `onecx_pipecat/pipeline/prompt.py` composes it with the existing system prompt.
- `tests/test_adapters.py` verifies response parsing.
- `tests/test_prompt.py` verifies order, trimming, and optional/blank behavior.
- `docs/modules/ROOT/pages/adapter-interfaces.adoc` documents the returned field and prompt composition.

## Task 1: Expose the additional prompt in ai-provider configuration

**Files:**
- Modify: `../onecx-ai-provider-svc/src/main/openapi/onecx-ai-v1.yaml`
- Modify: `../onecx-ai-provider-svc/src/main/java/org/tkit/onecx/ai/provider/rs/external/v1/mappers/AgentConfigurationMapper.java`
- Test: `../onecx-ai-provider-svc/src/test/java/org/tkit/onecx/ai/provider/rs/external/v1/mappers/AgentConfigurationMapperTest.java`

**Interfaces:**
- Consumes: `Agent.additionalPrompt` (optional persisted string).
- Produces: `AgentConfiguration.agent.additionalPrompt` (optional string in the `GET /v1/agents/{id}/configuration` response).

- [ ] **Step 1: Add a failing mapper assertion**

In `AgentConfigurationMapperTest.build_mapsTopLevelFields`, set
`agent.setAdditionalPrompt("Ask for tenant first.")` before calling
`mapper.build(...)`, then assert:

```java
assertThat(config.getAgent().getAdditionalPrompt()).isEqualTo("Ask for tenant first.");
```

- [ ] **Step 2: Run the focused mapper test and confirm it fails**

Run from `../onecx-ai-provider-svc`:

```bash
mvn -Dtest=AgentConfigurationMapperTest test
```

Expected: the response DTO does not yet expose the `additionalPrompt` getter/property.

- [ ] **Step 3: Add the optional response property and mapping**

Under `components.schemas.AgentConfigurationAgent.properties` in
`onecx-ai-v1.yaml`, add:

```yaml
additionalPrompt:
  type: string
```

In `AgentConfigurationMapper.agentObject(Agent agent)`, add the explicit
MapStruct mapping:

```java
@Mapping(target = "additionalPrompt", source = "agent.additionalPrompt")
```

Keep the existing `name` and `description` mappings. Do not make the field
required.

- [ ] **Step 4: Run the focused mapper test**

Run:

```bash
mvn -Dtest=AgentConfigurationMapperTest test
```

Expected: `AgentConfigurationMapperTest` passes and the generated response DTO
contains the stored value.

- [ ] **Step 5: Review and commit the service change**

Check `git diff --check`, review the diff to confirm the public response exposes
only the intended optional prompt field, then commit the service change:

```bash
git add src/main/openapi/onecx-ai-v1.yaml src/main/java/org/tkit/onecx/ai/provider/rs/external/v1/mappers/AgentConfigurationMapper.java src/test/java/org/tkit/onecx/ai/provider/rs/external/v1/mappers/AgentConfigurationMapperTest.java
git commit -m "feat: expose agent additional prompt in configuration"
```

Open the ai-provider service PR before starting the Pipecat PR; record its URL
for the Pipecat PR description.

## Task 2: Consume and compose the additional prompt in Pipecat

**Files:**
- Modify: `onecx_pipecat/models.py`
- Modify: `onecx_pipecat/adapters/agent_config.py`
- Modify: `onecx_pipecat/pipeline/prompt.py`
- Test: `tests/test_adapters.py`
- Test: `tests/test_prompt.py`
- Modify: `docs/modules/ROOT/pages/adapter-interfaces.adoc`

**Interfaces:**
- Consumes: optional `agent.additionalPrompt` from the configuration endpoint.
- Produces: `AgentConfigSnapshot.additional_prompt: str`, consumed by `compose_system_prompt(snapshot, ai_context)`.

- [ ] **Step 1: Add failing parser coverage**

Add `"additionalPrompt": "  Ask for tenant first.  "` to the `agent` entry in
`_AGENT_CONFIGURATION` in `tests/test_adapters.py`, then assert the parsed
snapshot has `snap.additional_prompt == "  Ask for tenant first.  "`. This
assertion verifies parsing preserves the raw configured value; prompt
composition is responsible for trimming it.

Add a second test with the field omitted from the agent object:

```python
async def test_agent_config_without_additional_prompt_defaults_empty():
    body = {**_AGENT_CONFIGURATION, "agent": {"id": "agent-1", "name": "Helper"}}

    snap = await _pin(200, body)

    assert snap.additional_prompt == ""
```

- [ ] **Step 2: Run the focused parser test and confirm it fails**

Run from the Pipecat repository root:

```bash
pytest tests/test_adapters.py::test_agent_config_parses_ai_provider_configuration -q
```

Expected: the snapshot has no `additional_prompt` attribute yet.

- [ ] **Step 3: Add the snapshot field and parse it**

Add `additional_prompt: str = ""` to `AgentConfigSnapshot` in
`onecx_pipecat/models.py`. In `_parse_configuration` in
`onecx_pipecat/adapters/agent_config.py`, read
`str(agent.get("additionalPrompt") or "")` into that field. Missing or null
values must remain empty.

- [ ] **Step 4: Add failing prompt-composition cases**

Update `_snap` in `tests/test_prompt.py` to accept
`additional_prompt: str = ""` and pass it to `AgentConfigSnapshot`. Add this
test to verify block ordering and trimming:

```python
def test_additional_prompt_follows_scaffold_and_precedes_skills_and_ai_context():
    skills = [SkillSnapshot(name="Lookup", instruction="lookup")]
    snapshot = _snap(
        "  be brief  ",
        skills,
        additional_prompt="  Ask for tenant first.  ",
    )

    assert compose_system_prompt(snapshot, ['{"page":"home"}']) == (
        "be brief\n\nAsk for tenant first.\n\n"
        "Available skills:\n- Lookup: Lookup\n"
        "Activate a relevant skill before applying its instructions.\n\n"
        'AI context:\n{"page":"home"}'
    )
```

Add a test asserting
`compose_system_prompt(_snap(additional_prompt=" \n ")) == "be brief"`.
Existing `_snap()` calls with no `additional_prompt` cover the missing-value
case; the combined test above proves nonblank prompt text is trimmed.

- [ ] **Step 5: Run the prompt tests and confirm the new cases fail**

Run:

```bash
pytest tests/test_prompt.py -q
```

Expected: the new tests fail because `compose_system_prompt` does not yet
include the snapshot's additional prompt.

- [ ] **Step 6: Compose the additional prompt**

In `compose_system_prompt` in `onecx_pipecat/pipeline/prompt.py`, append
`snapshot.additional_prompt.strip()` to the `blocks` after the scaffold prompt
and before the skills block, but only when nonblank. Keep the existing
`ai_context` block last and retain the current `"\n\n".join(blocks)` behavior.

- [ ] **Step 7: Run focused Pipecat tests and lint**

Run:

```bash
pytest tests/test_adapters.py tests/test_prompt.py -q
ruff check onecx_pipecat/models.py onecx_pipecat/adapters/agent_config.py onecx_pipecat/pipeline/prompt.py tests/test_adapters.py tests/test_prompt.py
```

Expected: all selected tests pass and Ruff reports no issues.

- [ ] **Step 8: Update adapter documentation**

In `docs/modules/ROOT/pages/adapter-interfaces.adoc`, document
`agent.additionalPrompt` among the parsed configuration fields and describe
that it follows `scaffold.systemPrompt` in the composed LLM system prompt.
Keep the documentation consistent with the skills and session context order.

- [ ] **Step 9: Review and commit the Pipecat change**

Check `git diff --check`, review the full diff, and commit only the implementation
files and docs:

```bash
git add onecx_pipecat/models.py onecx_pipecat/adapters/agent_config.py onecx_pipecat/pipeline/prompt.py tests/test_adapters.py tests/test_prompt.py docs/modules/ROOT/pages/adapter-interfaces.adoc
git commit -m "feat: use agent additional prompt in voice sessions"
```

Open the Pipecat PR with the service PR as a dependency and describe the
prompt-block ordering.

## Final verification and PR sequencing

- [ ] Confirm the service PR is open and its API field is `agent.additionalPrompt`.
- [ ] Confirm the Pipecat PR is open and references the service PR.
- [ ] Run the focused service and Pipecat commands after any review fixes.
- [ ] Verify `git status --short` in both repositories contains no unintended files.
