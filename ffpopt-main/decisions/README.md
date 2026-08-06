# Decisions — ffpopt

Lightweight ADRs (architectural decision records) for ffpopt. Each
decision is one Markdown file, dated, with status, context, decision, and
consequences.

This directory is for **small-to-medium** decisions — the kind that affect
how part of the repo works but don't warrant a full design doc. Larger
decisions (cross-repo contracts, multi-quarter migrations) belong in their
own document, referenced from `~/Work/claude-toolkit/META.md` if they cross
repo boundaries.

When in doubt, draft as a decision here and promote to a larger doc only if
it grows beyond a single-file scope.

## File naming

`NNNN-short-slug.md` where `NNNN` is a four-digit zero-padded sequence
number. First decision is `0001-`. Slugs are kebab-case, 3–6 words.

## Template

Copy `_template.md` in this directory as the starting point for a new
decision. Bump the number, fill in the fields, commit.

## Status semantics

- **Accepted** — currently in effect. The decision is treated as a frozen
  doc: do not edit in place; supersede if it changes.
- **Superseded by NNNN** — a later decision replaced this one. The replacing
  doc names the supersession explicitly and explains the change.
- **Reversed** — tried, didn't work, abandoned. Kept as a record so the
  reasoning isn't relitigated.

## When to supersede vs reverse

- If the new direction is a *refinement* of the old, supersede. The old
  reasoning was largely right; the new entry sharpens it.
- If the new direction *repudiates* the old, reverse. The old reasoning was
  wrong on its merits; the new entry explains why.

Reversal is rarer than supersession in a healthy repo. If you are reversing
often, the pre-decision discussion is too thin.

## Discoverability

`dev/claude-skills/historian/SKILL.md` should reference this directory under
its `Pointers` section so future Claude Code sessions find it when asked
"why is it this way."
