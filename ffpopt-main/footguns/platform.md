# ffpopt Platform Footguns

> Cross-cutting pitfalls specific to ffpopt. Add an entry when a
> problem wasn't obvious from the existing docs and isn't naturally scoped
> to one actor's `Anti-patterns` section.

<!--
Entry shape (copy from _template.md):

### Footgun title

A one-sentence description of the counterintuitive behavior.

- **Why it happens:** What makes the wrong path tempting or seemingly correct.
- **The right way:** What to do instead.
- **Where to learn more:** Doc reference, PR link, issue link, or commit.
-->

### Wavefront results are not bit-reproducible at `nproc > 1`

A relaxed-dihedral wavefront scan (`run_dihed_wavefront`, `ffpopt-DihedWavefront.py`,
and the twist workflow) can return slightly different per-angle minima between
otherwise identical runs when `nproc > 1`.

- **Why it happens:** the scan is a calculation queue, not a level barrier
  (`decisions/0001-wavefront-calculation-queue.md`). Worker results are folded in
  *completion order*, and a node only spawns neighbors when it improves its
  angle's running minimum — so which redundant nodes get explored depends on the
  order workers happen to finish in. The final minima are a minimization
  heuristic over the explored set, not a canonical value.
- **The right way:** treat wavefront energies as a heuristic minimum, not an
  exact, reproducible number. For a bit-reproducible single run, use `nproc=1`
  (serial, completion order is deterministic). Do not write tests or downstream
  logic that assume two `nproc>1` runs yield identical structures/energies.
- **Where to learn more:** `decisions/0001-wavefront-calculation-queue.md`;
  `WaveFront.Wavefront.calculate`; GitLab issue #30.
