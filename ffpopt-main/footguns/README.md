# Footguns — ffpopt

Counterintuitive things, common mistakes, anti-patterns. Each entry saves a
future engineer (or Claude session) from rediscovering pain.

## Split with `dev/claude-skills/<actor>/SKILL.md` Anti-patterns

Per-actor `Anti-patterns` sections in `dev/claude-skills/` are role-scoped —
one actor's checklist will surface them, and Rupert blocks on them per the
enforcement contract. `footguns/` captures **cross-cutting** pitfalls:
things a developer might trip on regardless of which role they're acting in,
or things that span multiple actors.

When in doubt:
- Fits cleanly inside one actor's domain → put it in that actor's `Anti-patterns`.
- Touches multiple domains, or is operational/cross-cutting → put it here.

Entries in this directory are advisory. They do not block PR review on their
own; they exist to spread knowledge and prevent recurrence.

## Per-entry format

```markdown
### Footgun title

A one-sentence description of the counterintuitive behavior.

- **Why it happens:** What makes the wrong path tempting or seemingly correct.
- **The right way:** What to do instead.
- **Where to learn more:** Doc reference, PR link, issue link, or commit.
```

See `_template.md` in this directory for a copy-paste starting point.

## When to add an entry

When you (or someone else) hits a problem that wasn't obvious from the
existing docs. The bar is low — saving one future debugging hour justifies
an entry. Better to over-document footguns than to lose the lesson.

## Organization

One file per major area (e.g., `platform.md`, `deploy.md`, `<integration>.md`).
Within a file, group entries under topical headers. If a file grows past
~150 lines, split it.
