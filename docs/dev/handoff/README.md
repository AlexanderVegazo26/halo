# Agent handoff copies

These are tracked copies of the orchestrator's session-scoped scratchpad files, kept so a
new session can resume even if the temp scratchpad is cleared.

- `prompts/`: each workstream's brief. `resume_*` are attempt-2 prompts, `resume2_*` are
  attempt-3 addenda, `orig_*` / `prompt_*` are originals.
  - Paths inside them that point at `...\AppData\Local\Temp\claude\...\scratchpad\` refer
    to the session that wrote them.
  - When relaunching an agent, point it at this directory instead, and give it a new
    status-file path.
- `status/`: each agent's last `status_<WS>.md`, refreshed at milestone commits.
- `agents.md`: the map from workstream to agent id. The ids are valid only in session
  d2c57405, the session that launched these agents.

The resume procedure is in `/memory.md`.
