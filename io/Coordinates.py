import warnings
from typing import Optional,  Union
import shutil
from pathlib import Path

import numpy as np

import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element, guess_masses


class Coordinates:
    """Thin MDAnalysis wrapper for reading and transforming structure coordinates."""

    def __init__(self, filename: Union[Path, str], filetype: str = 'pdb'):
        """
        Load a structure and sanitize masses for center-of-mass operations.

        Parameters
        ----------
        filename : Union[Path, str]
            Path to the structure file to read.
        filetype : str, optional
            File type hint (default: ``'pdb'``).
        """
        self.filename = Path(filename)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.u = mda.Universe(filename)
        self.original_coords = np.array(self.get_coordinates(), dtype=float, copy=True)

        # If the mol2 comes from antechaamber, then the atom names are weird and both rdkit and mda will have trouble
        if np.any(np.isclose(self.u.atoms.masses, 0, atol=0.1)):
            self.u.guess_TopologyAttrs(to_guess=['elements'], force_guess=['masses'])
        # We tried to get correct masses but may have failed in the process. Lack of masses will fail
        # MDAnalysis's center_of_mass(), so just set them to 1.0, since the exact values are not important
        self.u.atoms.masses[np.isclose(self.u.atoms.masses, 0, atol=0.1)] = 1.0

        return

    def get_coordinates(self):
        """Return the current atomic coordinates.

        Returns
        -------
        np.ndarray
            Coordinates of the atoms in the structure.
        """
        return self.u.atoms.positions

    def get_elements(self):
        """Return atomic element symbols.

        Returns
        -------
        list
            Element symbols for each atom.
        """
        try:
            return [atom.element for atom in self.u.atoms]
        except (AttributeError, TypeError, ValueError) as exc:
            import warnings
            warnings.warn(
                f"Could not read atom.element ({type(exc).__name__}: {exc}); "
                "guessing elements from atom names",
                stacklevel=2,
            )
            return self._get_elements_from_topology()

    def _get_elements_from_topology(self):
        """Guess element symbols from atom names in the topology.

        Returns
        -------
        list
            Guessed element symbols for each atom.
        """
        from MDAnalysis.topology.guessers import guess_types
        elements = guess_types(self.u.atoms.names)
        return elements

    def update_coordinates(self, coords, original=False):
        """Replace the current atomic coordinates.

        Parameters
        ----------
        coords : np.ndarray
            New coordinates with the same shape as the current positions.
        original : bool, optional
            If True, also update the stored original coordinates used by
            :meth:`rotate`.
        """
        assert np.shape(coords) == np.shape(self.get_coordinates()), "Coordinate dimensions do not match"
        self.u.atoms.positions = coords
        if original:
            self.original_coords = coords
        return

    def rotate(self, alpha=0.0, beta=0.0, gamma=0.0):
        """Rotate coordinates about the center of mass using Euler angles.

        Rotations are applied in order alpha (x), beta (y), gamma (z), matching
        the previous MDAnalysis ``rotateby`` sequence. Angles are in degrees.

        Parameters
        ----------
        alpha : float
            Rotation about the x-axis (degrees).
        beta : float
            Rotation about the y-axis (degrees).
        gamma : float
            Rotation about the z-axis (degrees).

        Returns
        -------
        np.ndarray
            Rotated coordinates with shape ``(n_atoms, 3)``.
        """
        coords = np.asarray(self.original_coords, dtype=float)
        # COM from original geometry (masses already sanitized in __init__)
        masses = self.u.atoms.masses
        com = np.average(coords, axis=0, weights=masses)

        a, b, g = np.deg2rad([alpha, beta, gamma])
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cg, sg = np.cos(g), np.sin(g)

        # Intrinsic/extrinsic composition matching sequential Rx, Ry, Rz on positions
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, ca, -sa],
                       [0.0, sa, ca]])
        ry = np.array([[cb, 0.0, sb],
                       [0.0, 1.0, 0.0],
                       [-sb, 0.0, cb]])
        rz = np.array([[cg, -sg, 0.0],
                       [sg, cg, 0.0],
                       [0.0, 0.0, 1.0]])
        rotation = rz @ ry @ rx

        rotated = (coords - com) @ rotation.T + com
        self.u.atoms.positions = rotated
        return rotated

    def rotate_matrix(self, rotation: np.ndarray) -> np.ndarray:
        """Rotate the original coordinates using an explicit rotation matrix.

        The rotation is applied about the mass-weighted center of mass. This
        is the path used by quaternion SO(3) orientation protocols.

        Parameters
        ----------
        rotation : np.ndarray, shape (3, 3)
            Proper orthogonal rotation matrix.

        Returns
        -------
        np.ndarray
            Rotated coordinates with shape ``(n_atoms, 3)``.

        Raises
        ------
        ValueError
            If ``rotation`` is not a valid proper rotation matrix.
        """
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"Expected rotation shape (3, 3), got {rotation.shape}")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("Rotation matrix must be orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("Rotation matrix must have determinant +1")

        coords = np.asarray(self.original_coords, dtype=float)
        com = np.average(coords, axis=0, weights=self.u.atoms.masses)
        rotated = (coords - com) @ rotation.T + com
        self.u.atoms.positions = rotated
        return rotated


def SimpleXYZ(file_obj, coordinates):
    """Write coordinates to a simple XYZ trajectory frame.

    Parameters
    ----------
    file_obj : file object
        Open file handle to write to.
    coordinates : np.ndarray
        Coordinates to write.
    """
    file_obj.write(f"{len(coordinates)}\n")
    file_obj.write("Generated by ligand_param\n")
    for i, coord in enumerate(coordinates):
        file_obj.write(f"{i + 1} {coord[0]} {coord[1]} {coord[2]}\n")
    return


class Mol2Writer:
    """Write an MDAnalysis Universe selection to a mol2 file."""

    def __init__(self, u, filename=None, selection="all"):
        """
        Parameters
        ----------
        u : MDAnalysis.Universe
            Universe to write.
        filename : str, optional
            Output mol2 path.
        selection : str, optional
            Atom selection string (default: ``'all'``).
        """
        self.u = u
        self.filename = Path(filename)
        self.selection = selection
        return

    def _write(self):
        """Write the selected atoms to mol2 via MDAnalysis."""
        ag = self.u.select_atoms(self.selection)
        ag.write(self.filename)

    def _remove_blank_lines(self):
        """Remove blank lines from the written mol2 file.

        Raises
        ------
        FileNotFoundError
            If the output file does not exist.
        """
        if Path(self.filename).exists():
            # Read the file and filter out blank lines
            with open(self.filename, 'r') as file:
                lines = file.readlines()
                non_blank_lines = [line for line in lines if line.strip()]

            # Write the non-blank lines back to the file
            with open(self.filename, 'w') as file:
                file.writelines(non_blank_lines)
        else:
            raise FileNotFoundError(f"File {self.filename} not found.")

    def write(self):
        """Write the mol2 file and strip blank lines that confuse antechamber."""
        self._write()
        self._remove_blank_lines()
        return


def _split_pdb_element_charge(token: str) -> tuple[str, str]:
    """Split Open Babel-style ``O1-`` / ``N+1`` into PDB element + charge.

    Parameters
    ----------
    token : str
        Trailing PDB element/charge field (columns 77-80 or a overflow token).

    Returns
    -------
    element : str
        One- or two-letter element symbol (``O``, ``Cl``, …).
    charge : str
        PDB charge field (``1-``, ``2+``, …) or empty.
    """
    import re

    t = (token or "").strip()
    if not t:
        return "", ""
    m = re.fullmatch(
        r"([A-Za-z]{1,2})(?:(\d+)([+-])|([+-])(\d+)|([+-]))?",
        t,
    )
    if not m:
        letters = "".join(c for c in t if c.isalpha())[:2]
        if not letters:
            return "", ""
        elem = letters[0].upper() + letters[1:].lower()
        return elem, ""
    elem = m.group(1)
    elem = elem[0].upper() + elem[1:].lower()
    if m.group(2) and m.group(3):
        charge = f"{m.group(2)}{m.group(3)}"
    elif m.group(4) and m.group(5):
        charge = f"{m.group(5)}{m.group(4)}"
    elif m.group(6):
        charge = f"1{m.group(6)}"
    else:
        charge = ""
    return elem, charge


def _rewrite_pdb_atom_line(line: str) -> str:
    """Rewrite ATOM/HETATM so cols 77-78 are element and 79-80 are charge."""
    nl = "\n" if line.endswith("\n") else ""
    raw = line.rstrip("\n")
    rec = raw[:6].strip()
    if rec not in ("ATOM", "HETATM"):
        return line
    if len(raw) < 80:
        raw = raw.ljust(80)
    elem, charge = _split_pdb_element_charge(raw[76:80])
    if not elem:
        # Open Babel often parks ``O1-`` after the B-factor instead of cols 77-80.
        elem, charge = _split_pdb_element_charge(raw[66:].strip())
    if not elem:
        atom_name = raw[12:16].strip()
        letters = "".join(c for c in atom_name if c.isalpha())[:2].upper()
        special = {"CL": "Cl", "BR": "Br", "NA": "Na", "MG": "Mg", "FE": "Fe", "ZN": "Zn"}
        if letters in special:
            elem = special[letters]
        elif letters:
            elem = letters[0]
    if not elem:
        elem = "C"
    return raw[:76] + f"{elem:>2}{charge:<2}" + nl


def _rewrite_pdb_conect_line(line: str) -> list[str]:
    """Keep CONECT, but drop duplicate partners (Open Babel S=O as ``15 15``)."""
    parts = line.split()
    if len(parts) < 2:
        return [line if line.endswith("\n") else line + "\n"]
    serial = int(parts[1])
    unique: list[int] = []
    for p in parts[2:]:
        try:
            idx = int(p)
        except ValueError:
            continue
        if idx not in unique:
            unique.append(idx)
    if not unique:
        return [f"CONECT{serial:5d}\n"]
    out = []
    for i in range(0, len(unique), 4):
        rec = f"CONECT{serial:5d}"
        for idx in unique[i : i + 4]:
            rec += f"{idx:5d}"
        out.append(rec + "\n")
    return out


def count_structure_atoms(path: Union[Path, str]) -> int:
    """Count atoms in a PDB (ATOM/HETATM) or mol2 (``@<TRIPOS>ATOM``) file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdb":
        n = 0
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    n += 1
        return n
    if suffix == ".mol2":
        n = 0
        in_atom = False
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("@<TRIPOS>"):
                    in_atom = line.startswith("@<TRIPOS>ATOM")
                    continue
                if in_atom and line.strip():
                    n += 1
        return n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(len(mda.Universe(str(p)).atoms))


def _element_from_mol2_name_type(name: str, atype: str) -> str:
    """Map mol2 atom name / GAFF type to a Gaussian element symbol."""
    for token in (atype, name):
        letters = "".join(c for c in (token or "") if c.isalpha())
        if not letters:
            continue
        two = letters[:2].capitalize()
        if two in {"Cl", "Br", "Na", "Mg", "Fe", "Zn", "Si"}:
            return two
        return letters[0].upper()
    return "C"


def parse_structure_atoms(path: Union[Path, str]) -> list[tuple[str, np.ndarray]]:
    """Read element symbols and coordinates from PDB or mol2 (no MDA).

    MDAnalysis element guessing can invent a hydrogen on anionic oxygen when
    building the Gaussian ``.com``. This parser only uses file records.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    atoms: list[tuple[str, np.ndarray]] = []
    if suffix == ".pdb":
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                rewritten = _rewrite_pdb_atom_line(line)
                elem = rewritten[76:78].strip() or "C"
                xyz = np.array(
                    [float(rewritten[30:38]), float(rewritten[38:46]), float(rewritten[46:54])],
                    dtype=float,
                )
                atoms.append((elem, xyz))
        return atoms
    if suffix == ".mol2":
        in_atom = False
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("@<TRIPOS>"):
                    in_atom = line.startswith("@<TRIPOS>ATOM")
                    continue
                if not in_atom or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                name, xs, ys, zs, atype = parts[1], parts[2], parts[3], parts[4], parts[5]
                elem = _element_from_mol2_name_type(name, atype)
                atoms.append((elem, np.array([float(xs), float(ys), float(zs)], dtype=float)))
        return atoms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(p))
        elems = [str(a.element) for a in u.atoms]
        return list(zip(elems, np.asarray(u.atoms.positions, dtype=float)))


def match_current_to_reference(
    reference: list[tuple[str, np.ndarray]],
    current: list[tuple[str, np.ndarray]],
    max_dist: float = 0.75,
) -> tuple[list[str], np.ndarray]:
    """Keep current atoms that match a reference, dropping extras (added H).

    Aligns on the heavy-atom centroid so a spurious OH hydrogen does not
    shift the frame. Returned coordinates are the current (e.g. centered)
    positions in reference order.
    """
    if len(current) < len(reference):
        raise RuntimeError(
            f"Current structure has fewer atoms ({len(current)}) than the "
            f"reference ({len(reference)})."
        )
    ref_e = [str(e) for e, _ in reference]
    ref_x = np.asarray([x for _, x in reference], dtype=float)
    cur_e = [str(e) for e, _ in current]
    cur_x = np.asarray([x for _, x in current], dtype=float)

    def _heavy_centroid(elems, xyz):
        heavy = np.array(
            [row for el, row in zip(elems, xyz) if str(el).upper() != "H"],
            dtype=float,
        )
        if heavy.size == 0:
            return xyz.mean(axis=0)
        return heavy.mean(axis=0)

    ref_a = ref_x - _heavy_centroid(ref_e, ref_x)
    cur_a = cur_x - _heavy_centroid(cur_e, cur_x)
    used: set[int] = set()
    keep: list[int] = []
    for e, x in zip(ref_e, ref_a):
        best_j, best_d = None, 1e9
        for j, (ej, y) in enumerate(zip(cur_e, cur_a)):
            if j in used or str(ej).upper() != str(e).upper():
                continue
            d = float(np.linalg.norm(x - y))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None or best_d > max_dist:
            raise RuntimeError(
                f"Could not match reference {e} atom onto the current "
                f"structure (best distance {best_d:.3f} A). "
                "The topology may have changed, not just an extra hydrogen."
            )
        used.add(best_j)
        keep.append(best_j)
    elems = [cur_e[j] for j in keep]
    coords = cur_x[keep]
    return elems, coords


def sanitize_pdb_ligand(src: Union[Path, str], dst: Union[Path, str]) -> Path:
    """Write an Amber-safe PDB without adding or removing atoms.

    Open Babel often writes the anionic sulfate oxygen as element ``O1-``
    and lists S=O twice in CONECT. Antechamber then treats that oxygen as
    an alcohol and appends a hydrogen (42-atom SDS -> 43-atom Gaussian).

    This keeps every ATOM/HETATM, rewrites element/charge into columns
    77-80, and uniquifies CONECT partners. The source file is not modified.
    """
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    with open(src_p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            key = line[:6].strip()
            if key in ("ATOM", "HETATM"):
                out_lines.append(_rewrite_pdb_atom_line(line))
            elif key == "CONECT":
                out_lines.extend(_rewrite_pdb_conect_line(line))
            else:
                out_lines.append(line if line.endswith("\n") else line + "\n")
    with open(dst_p, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)
    return dst_p


def Remove_PDB_CONECT(filename: Union[Path, str], backup: bool = False):
    """Deprecated: previously stripped CONECT, which protonated anions.

    Now sanitizes in place (keeps unique CONECT, fixes ``O1-`` elements).
    Prefer :func:`sanitize_pdb_ligand`, which does not modify the source.
    """
    fn = Path(filename)
    if backup:
        shutil.copyfile(fn, fn.parent / f"input_{fn.name}")
    tmp = fn.with_name(fn.stem + ".sanitized_tmp.pdb")
    sanitize_pdb_ligand(fn, tmp)
    tmp.replace(fn)
    return
