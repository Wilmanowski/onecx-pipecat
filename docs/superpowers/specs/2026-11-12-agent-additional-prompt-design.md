# Agent additional prompt in voice sessions

## Goal

Use the agent's configured `additionalPrompt` in Pipecat voice sessions. The
value is currently persisted by ai-provider, but the public
`GET /v1/agents/{id}/configuration` response does not include it, so Pipecat
cannot consume it.

## Design

Expose the optional string as `agent.additionalPrompt` in the external
ai-provider API contract. Map it from the stored `Agent.additionalPrompt` when
building the configuration response. Pipecat will parse that field into the
pinned `AgentConfigSnapshot` as `additional_prompt`.

Compose the LLM system prompt in this order:

1. The existing scaffold `systemPrompt`, if nonblank.
2. The agent `additionalPrompt`, if nonblank.
3. The available-skills block, if any.
4. The session-provided `ai_context`, if any.

Trim the two configured prompt strings and join all present blocks with a
blank line. Missing or blank `additionalPrompt` leaves the existing prompt
unchanged. The snapshot remains pinned for the session, matching the existing
agent configuration lifecycle.

## Scope and compatibility

- Update the ai-provider service's external OpenAPI schema, response mapping,
  and focused endpoint/mapping tests.
- Update Pipecat's snapshot model, response parser, prompt composer, tests, and
  adapter/prompt documentation.
- Keep the field optional so older or incomplete responses continue to work.
- Do not change protocol messages, session `ai_context`, skill handling, or
  unrelated runtime prompt behavior.
- Implement and publish the service and Pipecat changes as separate PRs, with
  the service response change available as a dependency for Pipecat.

## Verification

- Service tests verify the configuration endpoint exposes the stored
  `additionalPrompt` under `agent`.
- Pipecat tests verify it is parsed and included after `systemPrompt`, while
  absent/blank values do not add a prompt block.
- Run the smallest relevant service and Pipecat test selectors, plus each
  repository's required formatting or type checks if applicable.
