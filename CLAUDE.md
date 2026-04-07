# ChessGNN Session Notes

This repo is Vector's session root.

Keep repo-root session guidance minimal. Canonical local persona and workspace files live under `.openclaw/workspace-meta/` so the repo root stays clean.

Before doing substantive work, read and follow:
- `.openclaw/workspace-meta/AGENTS.md`
- `.openclaw/workspace-meta/SOUL.md`
- `.openclaw/workspace-meta/TOOLS.md`
- `.openclaw/workspace-meta/IDENTITY.md`
- `.openclaw/workspace-meta/USER.md`
- `.openclaw/workspace-meta/HEARTBEAT.md` when it exists

Do not recreate those persona files at the repo root unless the runtime forces it. If they reappear, treat them as local scaffolding rather than repository content.

Current expectations:
- Keep work scoped to this repository.
- Record the config, baseline, and verification method for experiment or model-facing changes.
- Treat regressions, unstable metrics, and data-quality issues as blockers.