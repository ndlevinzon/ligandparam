from __future__ import annotations

from collections import defaultdict
from pathlib import Path

try:
    import parmed  # noqa: F401
except ImportError:  # pragma: no cover
    parmed = None

try:
    import rdkit  # noqa: F401
except ImportError:  # pragma: no cover
    rdkit = None

from .fragments import build_candidate_fragments
from .io import load_ligand
from .models import CandidateFragment, FragmentConfig, FragmentationResult, InputBundle
from .optimize import select_fragments
from .screen import screen_candidate
from .torsions import enumerate_torsions, match_central_bond_smarts
from .writers import write_fragment_index, write_fragment_outputs, write_summary


def fragment_ligand(
    bundle: InputBundle,
    out_dir: Path,
    config: FragmentConfig,
) -> FragmentationResult:
    """Run the full ligand fragmentation workflow for one input bundle.

    Args:
        bundle: Input ligand file bundle to load and validate.
        out_dir: Output directory where fragment artifacts should be written.
        config: Fragmentation and screening configuration.

    Returns:
        A :class:`FragmentationResult` describing the selected fragments and
        any rejected torsions.
    """

    ligand = load_ligand(bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if parmed is None:
        warnings.append("ParmEd not installed; using pure-Python parser/writer fallback.")
    if rdkit is None:
        warnings.append("RDKit not installed; using pure-Python torsion heuristics.")

    torsions = enumerate_torsions(
        ligand,
        include_rigid_single_bonds=config.include_rigid_single_bonds,
        rotatable_bond_smarts=config.rotatable_bond_smarts,
    )
    torsion_map = {torsion.label: torsion for torsion in torsions}
    candidate_pool: dict[str, CandidateFragment] = {}
    rejected_torsions: dict[str, object] = {}

    # When restrict_to_bond_smarts is set, keep only the torsions whose central
    # bond matches one of the allow-list patterns; the rest are dropped before
    # any fragment is built and reported as rejected. Restriction composes with
    # rotatable_bond_smarts (which nominates extra bonds): nomination decides
    # what is rotatable, restriction decides which of those to actually fit.
    if config.restrict_to_bond_smarts:
        allowed_bonds = match_central_bond_smarts(
            ligand, config.restrict_to_bond_smarts
        )
        kept_torsions = []
        for torsion in torsions:
            if torsion.bond in allowed_bonds:
                kept_torsions.append(torsion)
            else:
                rejected_torsions[torsion.label] = "excluded_by_restrict_to_bond_smarts"
        if not kept_torsions:
            warnings.append(
                "restrict_to_bond_smarts matched no rotatable torsions; "
                "no fragments will be generated."
            )
        torsions = kept_torsions
        torsion_map = {torsion.label: torsion for torsion in torsions}
    accepted_by_torsion: dict[str, list[CandidateFragment]] = {}
    for torsion in torsions:
        valid_for_torsion = False
        best_failure: tuple[float, dict[str, object]] | None = None
        accepted_candidates: list[CandidateFragment] = []
        for candidate in build_candidate_fragments(
            ligand,
            torsion,
            include_rigid_single_bonds=config.include_rigid_single_bonds,
            rotatable_bond_smarts=config.rotatable_bond_smarts,
        ):
            candidate_pool.setdefault(candidate.candidate_id, candidate)
            screen = screen_candidate(
                ligand,
                torsion,
                candidate,
                angle_step=config.angle_step,
                thresholds=config.clash_thresholds,
            )
            counts_as_success = screen.accepted and (
                not candidate.is_parent_fallback or config.use_parent_fallback
            )
            if counts_as_success:
                pooled = candidate_pool[candidate.candidate_id]
                pooled.torsion_labels.update(candidate.torsion_labels)
                pooled.accepted_torsions.add(torsion.label)
                pooled.worst_margin = (
                    screen.worst_margin
                    if pooled.worst_margin is None
                    else min(pooled.worst_margin, screen.worst_margin)
                )
                accepted_candidates.append(pooled)
                valid_for_torsion = True
            elif screen.violation is not None and not candidate.is_parent_fallback:
                failure_payload = {
                    "reason": screen.reason,
                    "candidate_id": candidate.candidate_id,
                    "retained_atom_count": len(candidate.retained_atoms),
                    "cut_bonds": [list(bond) for bond in sorted(candidate.cut_bonds)],
                    "violation": screen.violation,
                }
                if best_failure is None or screen.worst_margin > best_failure[0]:
                    best_failure = (screen.worst_margin, failure_payload)
        if not valid_for_torsion:
            rejected_torsions[torsion.label] = (
                best_failure[1] if best_failure is not None else "no_valid_fragment_found"
            )
        else:
            accepted_by_torsion[torsion.label] = accepted_candidates

    preferred_candidate_ids: set[str] = set()
    for torsion_label, candidates in accepted_by_torsion.items():
        preferred = min(
            candidates,
            key=lambda candidate: (
                -candidate.ring_cap_count,
                len(candidate.retained_atoms),
                candidate.ring_count,
                -(candidate.worst_margin if candidate.worst_margin is not None else -999.0),
                len(candidate.cut_bonds),
                candidate.candidate_id,
            ),
        )
        preferred_candidate_ids.add(preferred.candidate_id)

    selected_candidates, covered_torsions, uncovered = select_fragments(
        candidate_pool,
        [torsion.label for torsion in torsions if torsion.label not in rejected_torsions],
        allow_parent_fallback=config.use_parent_fallback,
        preferred_candidate_ids=preferred_candidate_ids,
    )
    for torsion_label in sorted(uncovered):
        rejected_torsions.setdefault(torsion_label, "no_valid_fragment_found")

    assigned: dict[str, list[str]] = defaultdict(list)
    for torsion_label, candidate_id in covered_torsions.items():
        assigned[candidate_id].append(torsion_label)

    selected_fragments = [
        write_fragment_outputs(
            ligand,
            candidate,
            f"fragment_{index}",
            sorted(assigned[candidate.candidate_id]),
            torsion_map,
            out_dir,
            config,
            warnings,
        )
        for index, candidate in enumerate(selected_candidates, start=1)
    ]
    fragment_index_path = write_fragment_index(selected_fragments, out_dir)
    summary_path = write_summary(
        ligand,
        selected_fragments,
        covered_torsions,
        rejected_torsions,
        warnings,
        out_dir,
    )
    output_paths = {fragment.fragment_id: str(fragment.manifest_path.parent) for fragment in selected_fragments}
    return FragmentationResult(
        selected_fragments=selected_fragments,
        covered_torsions=covered_torsions,
        rejected_torsions=rejected_torsions,
        warnings=warnings,
        output_paths=output_paths,
        summary_path=summary_path,
        fragment_index_path=fragment_index_path,
    )
