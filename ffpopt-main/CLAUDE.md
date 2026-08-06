# CLAUDE.md — ffpopt

> Entry-point navigation for Claude Code working in this repo. Keep this file
> short and current — its job is to point at the right doc, not duplicate it.

## What this repo is

<one-paragraph description: what ffpopt does, who consumes it, where
it sits in the ATTMOS ecosystem. If unsure, see `claude-toolkit/META.md`>

## First things to read

1. `dev/claude-skills/INDEX.md` — available skill roles and when to consult each.
2. This file.
3. `README.md` — human-facing repo overview.
4. `GLOSSARY.md` — terminology specific to this repo.
5. `~/Work/claude-toolkit/META.md` — ATTMOS ecosystem map: what repos depend
   on this one, what this one depends on, what contracts cross the boundary.

## How to navigate

| Looking for... | Start here |
|---|---|
| Skill role to invoke for this task | `dev/claude-skills/INDEX.md` |
| Architecture and structure | `dev/claude-skills/architect/SKILL.md` |
| Coding conventions, naming, error handling | `dev/claude-skills/techlead/SKILL.md` |
| Tests, security, code quality | `dev/claude-skills/inspector/SKILL.md` |
| Deploy, CI/CD, infra, secrets | `dev/claude-skills/operator/SKILL.md` |
| Why a past decision was made | `dev/claude-skills/historian/SKILL.md`, then `decisions/` |
| External APIs, SDKs, webhooks | `dev/claude-skills/integrator/SKILL.md` |
| Docs build/publish | `dev/claude-skills/librarian/SKILL.md` |
| Domain term meaning | `GLOSSARY.md` |
| Repo-level decision record | `decisions/` |
| Cross-cutting pitfalls | `footguns/` |
| Cross-repo contracts | `~/Work/claude-toolkit/META.md` |
| ATTMOS Claude Code practices | `~/Work/claude-toolkit/practices/claude-code-practices.md` |

## How skills and guardrails interact

The `dev/claude-skills/` library defines **how to act** — which perspective to
take (architect, techlead, inspector, etc.) and which workflow that role
follows. The repo-root guardrails (`GLOSSARY.md`, `decisions/`, `footguns/`,
and this file) describe **what is in this repo** — its terminology, its
historical decisions, its known pitfalls.

A typical session picks a skill from `dev/claude-skills/INDEX.md` first, then
draws repo-specific context from the guardrails as needed. Skills are
enforced by Rupert on PR review; guardrails are advisory.

## Living vs frozen docs

Some docs evolve continuously with the code; others freeze once written.
Mixing the two is a footgun in itself.

**Living** (update in the same PR as related code changes):
- `README.md`, `CLAUDE.md` (this file), `GLOSSARY.md`
- `footguns/*`
- Every `dev/claude-skills/<actor>/SKILL.md` — required to update in the same
  PR that changes what the skill describes (see `practices/claude-code-practices.md`)

**Frozen** (don't edit; supersede by writing a new doc):
- `decisions/` entries with `Status: Accepted` — supersede via a later entry
  that references the prior one ("Supersedes 0042")
- Any sub-anchor or RFC marked shipped/closed

"Frozen" does not mean "never touch." It means: if the conclusion changes,
write a new doc that explicitly supersedes the old. The old stays as a record
of what was true at the time.

## What NOT to do without explicit confirmation

- Do not modify a `dev/claude-skills/<actor>/SKILL.md` file in a PR that is
  not also changing the thing the skill describes. Skills update *with*
  related code, not on their own.
- Do not delete a `decisions/` entry. Supersede with a new entry instead.
- Do not introduce new top-level directories without checking `README.md` and
  this file first.
- Do not silently change cross-references. If a target file moves, update
  every reference in the same PR.
- Do not paste customer data, credentials, or secrets into Claude — see
  `~/Work/claude-toolkit/practices/claude-code-practices.md` §7.

## Verification before merge

Before any PR merges, confirm:

1. Skills affected by the change have been updated in this PR (Rupert will
   flag if not, but catch it earlier).
2. Cross-references resolve — paths cited in changed files actually exist.
3. If a cross-repo contract changed, `~/Work/claude-toolkit/META.md` reflects
   the new shape.
4. If a new convention or pitfall was discovered during the work, a glossary
   entry or footgun entry captures it.
