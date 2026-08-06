#!/usr/bin/env bash
#
# sync-claude-skills.sh
#
# Symlinks each actor skill in this repo's dev/claude-skills/<actor>/ into
# ~/.claude/skills/<skill-name>/ (ONE symlink per actor). Claude Code only
# discovers skills at ~/.claude/skills/<name>/SKILL.md (one level deep, no
# recursion), so a single parent-directory symlink does NOT work.
#
# <skill-name> comes from each SKILL.md `name:` frontmatter (fallback
# <reponame>-<actor>). Source of truth = dev/claude-skills/ in the repo; the
# entries under ~/.claude/skills are relative symlinks, never copies.
#
# Usage:
#   ./dev/claude-skills/sync-claude-skills.sh            # link (default)
#   ./dev/claude-skills/sync-claude-skills.sh --unlink   # remove this repo's links
#   ./dev/claude-skills/sync-claude-skills.sh --status   # show current state
#   ./dev/claude-skills/sync-claude-skills.sh --name foo # override fallback slug
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || { echo "error: not inside a git repo" >&2; exit 1; }
SRC="$REPO_ROOT/dev/claude-skills"
[[ -d "$SRC" ]] || { echo "error: $SRC does not exist" >&2; exit 1; }

resolve_path() { python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
relpath()      { python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$2"; }
SRC_RESOLVED="$(resolve_path "$SRC")"

MODE="link"; NAME_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --unlink) MODE="unlink"; shift ;;
    --status) MODE="status"; shift ;;
    --name)   (( $# >= 2 )) || { echo "error: --name requires a value" >&2; exit 2; }
              NAME_OVERRIDE="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$NAME_OVERRIDE" ]]; then
  REPO_NAME="$NAME_OVERRIDE"
else
  REMOTE_URL="$(git -C "$REPO_ROOT" config --get remote.origin.url 2>/dev/null || true)"
  [[ -n "$REMOTE_URL" ]] && REPO_NAME="$(basename -s .git "$REMOTE_URL")" || REPO_NAME="$(basename "$REPO_ROOT")"
fi
DST_ROOT="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"

# read `name:` from the first YAML frontmatter block of a SKILL.md
skill_name_of() {
  awk 'BEGIN{n=0} /^---[[:space:]]*$/{n++; next}
       n==1 && /^name:/{sub(/^name:[[:space:]]*/,""); gsub(/[[:space:]]+$/,""); print; exit}' "$1"
}

# emit "<skillname>\t<actordir>" for each actor dir that has a SKILL.md
actor_links() {
  local d md name
  for d in "$SRC"/*/; do
    md="$d/SKILL.md"; [[ -f "$md" ]] || continue
    name="$(skill_name_of "$md")"; [[ -n "$name" ]] || name="${REPO_NAME}-$(basename "$d")"
    printf '%s\t%s\n' "$name" "${d%/}"
  done
  return 0
}

# symlinks in DST_ROOT that resolve exactly to SRC = broken parent links
legacy_parent_links() {
  local e; [[ -d "$DST_ROOT" ]] || return 0
  for e in "$DST_ROOT"/*; do
    [[ -L "$e" ]] || continue
    [[ "$(resolve_path "$e" 2>/dev/null || true)" == "$SRC_RESOLVED" ]] && printf '%s\n' "$e"
  done
  return 0
}

case "$MODE" in
  status)
    echo "repo:   $REPO_NAME"; echo "source: $SRC"; echo "target: $DST_ROOT/"
    while IFS=$'\t' read -r name dir; do
      link="$DST_ROOT/$name"
      if [[ -L "$link" && "$(resolve_path "$link")" == "$(resolve_path "$dir")" ]]; then
        echo "  [linked ✓] $name -> $(readlink "$link")"
      elif [[ -e "$link" ]]; then echo "  [conflict] $name (exists, not our link)"
      else echo "  [missing ] $name"; fi
    done < <(actor_links)
    legacy="$(legacy_parent_links)"
    [[ -n "$legacy" ]] && { echo "legacy parent symlink(s) (broken layout — removed on link):";
      while IFS= read -r e; do echo "  $e -> $(readlink "$e")"; done <<< "$legacy"; }
    ;;
  link)
    mkdir -p "$DST_ROOT"
    legacy="$(legacy_parent_links)"
    [[ -n "$legacy" ]] && while IFS= read -r e; do rm "$e"; echo "removed legacy parent symlink: $e"; done <<< "$legacy"
    while IFS=$'\t' read -r name dir; do
      link="$DST_ROOT/$name"
      if [[ -L "$link" ]]; then
        if [[ "$(resolve_path "$link")" == "$(resolve_path "$dir")" ]]; then echo "already linked: $name"; continue; fi
        echo "error: $link -> $(readlink "$link") — refusing to overwrite" >&2; continue
      fi
      [[ -e "$link" ]] && { echo "error: $link exists and is not a symlink — skipping" >&2; continue; }
      rel="$(relpath "$dir" "$DST_ROOT")"; ln -s "$rel" "$link"; echo "linked: $name -> $rel"
    done < <(actor_links)
    echo; echo "Restart Claude Code (or /reload-skills) to pick up the skills."
    ;;
  unlink)
    while IFS=$'\t' read -r name dir; do
      link="$DST_ROOT/$name"
      [[ -L "$link" && "$(resolve_path "$link")" == "$(resolve_path "$dir")" ]] && { rm "$link"; echo "unlinked: $name"; }
    done < <(actor_links)
    legacy="$(legacy_parent_links)"
    [[ -n "$legacy" ]] && while IFS= read -r e; do rm "$e"; echo "removed legacy parent symlink: $e"; done <<< "$legacy"
    ;;
esac

exit 0
