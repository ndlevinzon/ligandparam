# dev/claude-skills/

This directory is ffpopt's institutional memory for Claude Code. Each subdirectory is an "actor" — a perspective on the codebase (architect, techlead, operator, etc.) — with its own `SKILL.md` describing what someone in that role needs to know.

## For new contributors

After cloning, run the sync script once:

```bash
chmod +x ./dev/claude-skills/sync-claude-skills.sh
./dev/claude-skills/sync-claude-skills.sh
```

This creates **one symlink per actor** under `~/.claude/skills/` — e.g. `~/.claude/skills/ffpopt-architect → dev/claude-skills/architect` — named from each `SKILL.md`'s `name:` field. Claude Code only discovers skills one level deep, at `~/.claude/skills/<name>/SKILL.md`, so a single parent-directory symlink would not work. You only need to do this once per machine per repo; **restart Claude Code afterwards** to load the skills. (If an old single-directory `~/.claude/skills/ffpopt` link from a previous version is present, the script removes it automatically.)

Check status any time: `./dev/claude-skills/sync-claude-skills.sh --status`

## What's here

- `INDEX.md` — Claude-facing index of all actors and when to consult each
- `<actor>/SKILL.md` — one per actor, dense and Claude-readable
- `SKILLS_CONTEXT.md` — repo-specific input used during skill generation
- `sync-claude-skills.sh` — per-actor symlink script (run once after cloning)

See [INDEX.md](INDEX.md) for the actor list and what each covers.

## Maintenance

These files are *living documentation*. The cardinal rule:

> **PRs that change anything a skill describes must update that skill in the same PR.**

If you change architecture, add a convention, deprecate a pattern, or change deploy/CI behavior, the relevant skill file updates in the same PR. Reviewers will block PRs that don't.

Each skill has an owner listed in its footer. Owners re-read quarterly and update the `Last reviewed:` timestamp.

## Rupert review

PRs in this repo can be reviewed by Rupert (assign Rupert's GitHub account as a reviewer). Rupert reads these skills and flags deviations between the diff and the documented conventions. Two ways to resolve a Rupert comment:

1. Update the code to match the skill.
2. Update the skill if the new behavior is intentional and the convention is changing.

Either is fine. What's not fine is ignoring Rupert and merging — that's how the skills get out of sync with reality.

## Generating / regenerating

The skills were initially generated using `claude-toolkit`. To regenerate:

```bash
cd ~/Work/claude-toolkit
./scripts/generate.sh ~/Work/ffpopt --output-dir /tmp/skills-staging
diff -r ./dev/claude-skills/ /tmp/skills-staging/
# review and merge selectively
```

Day-to-day, prefer hand-editing skills in the PR that motivates the change.
