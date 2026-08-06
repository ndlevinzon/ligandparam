# SKILLS_CONTEXT.md — ffpopt

*Tiny per-repo input for `GENERATE_SKILLS.md`. Keep this short. If you can't fill a field confidently, leave it blank — Claude will detect or flag it in Gaps.*

## Identity

- **Repo name:** ffpopt
- **Skill prefix:** ffpopt
- **Primary owner (GitHub handle):** piskuliche
- **GitHub org/repo:** lbsr/ffpopt

## One-line description


FFPOPT is a bespoke force field parameterization engine for amber. 

## Known constraints Claude would otherwise miss


- This repo has been used for a paper, so major breaking changes should be considered carefully.
- This repo is hosted on gitlab and not github.

## Pointers that save a research pass

- Package root: `src/python/`
- Docs build (Sphinx): `docs/API`, `docs/UserDocs`
- Wavefront Source: `src/python/lib/WaveFront.py`
- Interfaces: `src/python/bin`

## Explicit non-goals for the skills

None -- generate all eight actors(User, Architect, TechLead, Inspector, Librarian, Operator, Historian, Integrator) to match the rag-tools coverage. If any actor turns out thing document what actually there and flag the gap rather than skipping the file. 


---

*When this file is filled in, invoke:*

```bash
claude -p "Read GENERATE_SKILLS.md and SKILLS_CONTEXT.md, then execute." \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose \
  > "claude-skills-$(date +%Y%m%d-%H%M%S).log" 2>&1 &
```
