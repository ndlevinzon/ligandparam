# ffpopt Skills — Index

Institutional memory for Claude Code working on ffpopt. Each file answers a
specific slice of "what do I need to know to act correctly here."

| Actor | When to consult | File |
|-------|-----------------|------|
| User | You're using ffpopt's CLI scripts or Python API, not modifying them. | [user/SKILL.md](user/SKILL.md) |
| Architect | You need to understand structure before changing it. | [architect/SKILL.md](architect/SKILL.md) |
| TechLead | You're writing or reviewing code. | [techlead/SKILL.md](techlead/SKILL.md) |
| Inspector | You're touching tests, security, or code-quality gates. | [inspector/SKILL.md](inspector/SKILL.md) |
| Librarian | You're writing, building, or publishing docs. | [librarian/SKILL.md](librarian/SKILL.md) |
| Operator | You're changing deploy, CI/CD, build, or environment. | [operator/SKILL.md](operator/SKILL.md) |
| Historian | You're about to reinvent something or question a weird choice. | [historian/SKILL.md](historian/SKILL.md) |
| Integrator | You're touching an external API, SDK, or ML model integration. | [integrator/SKILL.md](integrator/SKILL.md) |

## How Claude should use these
Consult the relevant skill(s) before acting on a task in its domain. Multi-domain tasks consult multiple skills. If a skill is silent on something, say so — don't invent a convention.

## How devs maintain these
- PRs that change anything a skill describes must update that skill in the same PR.
- Every skill has a single owner (GitHub handle in footer). Owners re-read quarterly and update `Last reviewed`.
- New conventions go in the relevant skill *before* being applied in code.
- If you find a convention that isn't in a skill, add it (and flag the PR that introduced it).
