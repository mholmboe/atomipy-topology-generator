"""Tests for the topology interchange layer. Runnable directly or via pytest."""
import os
import tempfile

import atomipy.import_topology as rd
import atomipy.write_topology as wr
from atomipy.topology import (Topology, Box, Atom, AtomType, Bond, Angle,
                              functional_forms as ff, units as U)


def _two_oh_system():
    """Two O–H bonds, SAME atom types (Oh/Ho) but DIFFERENT params — the per-site
    uniqueness fixture."""
    top = Topology()
    top.box = Box.from_box_dim([2.0, 2.0, 2.0])
    top.atom_types = [
        AtomType("Oh", element="O", atomic_number=8, mass=15.999,
                 lj={"sigma": 0.317, "epsilon": 0.65}),
        AtomType("Ho", element="H", atomic_number=1, mass=1.008,
                 lj={"sigma": 0.0, "epsilon": 0.0}),
    ]
    top.atoms = [
        Atom(0, type="Oh", element="O", mass=15.999, charge=-0.95, position=[0.0, 0, 0]),
        Atom(1, type="Ho", element="H", mass=1.008, charge=0.425, position=[0.097, 0, 0]),
        Atom(2, type="Oh", element="O", mass=15.999, charge=-0.95, position=[1.0, 0, 0]),
        Atom(3, type="Ho", element="H", mass=1.008, charge=0.425, position=[1.097, 0, 0]),
    ]
    top.bonds = [
        Bond(0, 1, "harmonic", {"b0": 0.0974, "k": 462750.0}),
        Bond(2, 3, "harmonic", {"b0": 0.0969, "k": 451000.0}),   # different!
    ]
    top.angles = []
    return top


def test_units_and_prefactor():
    p = ff.to_backend("bond", "harmonic", {"b0": 0.0974, "k": 462750.0}, "lammps_real")
    # b0: nm -> Å (x10); k: kJ/nm² -> kcal/Å² (/4.184/100) then /2 prefactor
    assert abs(p["b0"] - 0.974) < 1e-9, p
    assert abs(p["k"] - 462750.0 / (4.184 * 100 * 2)) < 1e-6, p
    # GROMACS keeps the ½ -> no /2, no unit change
    g = ff.to_backend("bond", "harmonic", {"b0": 0.0974, "k": 462750.0}, "gromacs")
    assert abs(g["k"] - 462750.0) < 1e-9 and abs(g["b0"] - 0.0974) < 1e-12
    # round-trip identity
    back = ff.from_backend("bond", "harmonic", p, "lammps_real")
    assert abs(back["k"] - 462750.0) < 1e-3 and abs(back["b0"] - 0.0974) < 1e-9
    print("PASS units_and_prefactor")


def test_json_roundtrip():
    top = _two_oh_system()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.json")
        wr.write_json(top, p)
        top2 = rd.read_json(p)
    assert top2.n_atoms == 4
    assert len(top2.bonds) == 2
    assert top.to_dict() == top2.to_dict(), "JSON round-trip not identity"
    print("PASS json_roundtrip")


def test_per_site_uniqueness_itp():
    top = _two_oh_system()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.itp")
        wr.write_itp(top, p)
        text = open(p).read()
        top2 = rd.read_itp(p)
    # both distinct b0 values appear inline
    assert "9.740000e-02" in text and "9.690000e-02" in text, text
    assert len(top2.bonds) == 2
    p1 = top2.bonds[0].params; p2 = top2.bonds[1].params
    assert abs(p1["b0"] - 0.0974) < 1e-6 and abs(p2["b0"] - 0.0969) < 1e-6
    assert abs(p1["k"] - 462750.0) < 1e-2 and abs(p2["k"] - 451000.0) < 1e-2
    print("PASS per_site_uniqueness_itp (inline, round-trips)")


def test_per_site_uniqueness_data():
    top = _two_oh_system()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.data")
        wr.write_data(top, p, units="real")
        text = open(p).read()
        top2 = rd.read_data(p, units="real")
    assert "2 bond types" in text, text
    # two distinct Bond Coeffs lines
    coeff_lines = [l for l in text.splitlines()
                   if l and l[0].isdigit() and "harmonic" in l]
    assert len(coeff_lines) == 2, coeff_lines
    # read back: two bonds, params recovered to canonical (different)
    b0s = sorted(round(b.params["b0"], 4) for b in top2.bonds)
    assert b0s == [0.0969, 0.0974], b0s
    print("PASS per_site_uniqueness_data (two types, round-trips)")


def test_cross_format_structural():
    top = _two_oh_system()
    with tempfile.TemporaryDirectory() as d:
        wr.write_itp(top, os.path.join(d, "a.itp"))
        t1 = rd.read_itp(os.path.join(d, "a.itp"))
        # carry positions + box across for .data (the .itp has neither; in a real
        # workflow they come from the companion .gro)
        t1.box = top.box
        for a, src in zip(t1.atoms, top.atoms):
            a.position = src.position
        wr.write_data(t1, os.path.join(d, "a.data"), units="real")
        t2 = rd.read_data(os.path.join(d, "a.data"), units="real")
    assert (t1.n_atoms, len(t1.bonds)) == (t2.n_atoms, len(t2.bonds)) == (4, 2)
    print("PASS cross_format_structural (itp->data->read preserves counts)")


def test_adapter_roundtrip():
    import atomipy as ap, glob
    f = glob.glob(os.path.join(os.path.dirname(ap.__file__),
                  "structures/minerals/UC_conf/Kaolinite*.pdb"))[0]
    atoms, box = ap.import_auto(f)
    top = rd.from_atoms_box(atoms, box)
    assert top.n_atoms == len(atoms)
    assert top.box is not None and top.has_positions()
    atoms2, box2 = wr.to_atoms_box(top)
    assert len(atoms2) == len(atoms)
    # box cell params round-trip (Å)
    for x, y in zip(box, box2):
        assert abs(float(x) - float(y)) < 1e-3, (box, box2)
    # positions round-trip Å->nm->Å
    a0, b0 = atoms[0], atoms2[0]
    assert abs(b0["x"] - a0["x"]) < 1e-4 and abs(b0["z"] - a0["z"]) < 1e-4
    print(f"PASS adapter_roundtrip ({top.n_atoms} atoms, box {[round(v,3) for v in box2]})")


def test_typed_itp_resolution():
    """A .itp whose [bonds] lack inline params but a [bondtypes] table exists —
    params must be resolved by atom type."""
    itp = """
[ atomtypes ]
 Oh 8 15.999 -0.95 A 0.317 0.65
 Ho 1 1.008  0.42 A 0.0   0.0
[ moleculetype ]
 MOL 1
[ atoms ]
 1 Oh 1 MIN O1 1 -0.95 15.999
 2 Ho 1 MIN H1 1  0.42  1.008
[ bondtypes ]
 Oh Ho 1 0.0974 462750.0
[ bonds ]
 1 2 1
"""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "typed.itp")
        open(p, "w").write(itp)
        top = rd.read_itp(p, validate=False)
    assert len(top.bonds) == 1
    b = top.bonds[0]
    assert b.params and abs(b.params["b0"] - 0.0974) < 1e-6 and abs(b.params["k"] - 462750.0) < 1e-2, b.params
    print("PASS typed_itp_resolution (bondtypes lookup by atom type)")


def _theta0_from_itp(path):
    import re
    vals, insec = [], False
    for ln in open(path):
        s = ln.split(";")[0].strip()
        if s.startswith("["):
            insec = re.match(r"\[\s*angles\s*\]", s, re.I) is not None
            continue
        if insec and s:
            t = s.split()
            if len(t) >= 5:
                try:
                    vals.append(round(float(t[4]), 2))
                except ValueError:
                    pass
    return vals


def test_harmonize_parity_with_legacy():
    """reduce.harmonize must reproduce the legacy harmonize_angles collapse
    (≈103 distinct θ0 -> ≈12), matching legacy's distinct-θ0 count."""
    import importlib, glob
    import atomipy as ap
    from atomipy.topology import reduce as red
    legacy = importlib.import_module("atomipy.write_top")

    f = glob.glob(os.path.join(os.path.dirname(ap.__file__),
                  "structures/minerals/UC_conf/Kaolinite*.pdb"))[0]
    atoms, Box = ap.import_auto(f)
    atoms, Box, _ = ap.replicate_system(atoms, Box, replicate=[2, 2, 1])
    with tempfile.TemporaryDirectory() as d:
        A = os.path.join(d, "explicit.itp"); B = os.path.join(d, "harm.itp")
        legacy.itp(atoms, Box=Box, file_path=A, explicit_bonds=1, explicit_angles=1)
        legacy.itp(atoms, Box=Box, file_path=B, explicit_bonds=1, explicit_angles=1,
                   detect_bimodal=True, harmonize_angles=True)
        top = rd.read_itp(A, validate=False)
        explicit_n = len({round(a.params["theta0"], 2) for a in top.angles if "theta0" in a.params})
        red.harmonize(top, categories=("angle",), detect_bimodal=True, threshold=30.0)
        new_n = len({round(a.params["theta0"], 2) for a in top.angles if "theta0" in a.params})
        legacy_n = len(set(_theta0_from_itp(B)))

    assert explicit_n > new_n, f"no collapse: {explicit_n} -> {new_n}"
    assert new_n == legacy_n, f"parity mismatch: new {new_n} vs legacy {legacy_n}"
    print(f"PASS harmonize_parity (explicit {explicit_n} distinct θ0 -> harmonized {new_n}; "
          f"legacy harmonized {legacy_n})")


def test_average_by_type_collapse():
    """average_by_type collapses to one parameter set per atom-type tuple, so
    extract_types yields ~one type per tuple (the legacy lmp() default)."""
    import glob
    import atomipy as ap
    from atomipy.topology import reduce as red, typing as ttyp
    f = glob.glob(os.path.join(os.path.dirname(ap.__file__),
                  "structures/minerals/UC_conf/Kaolinite*.pdb"))[0]
    atoms, Box = ap.import_auto(f)
    atoms, Box, _ = ap.replicate_system(atoms, Box, replicate=[2, 2, 1])
    with tempfile.TemporaryDirectory() as d:
        A = os.path.join(d, "explicit.itp")
        __import__("importlib").import_module("atomipy.write_top").itp(
            atoms, Box=Box, file_path=A, explicit_bonds=1, explicit_angles=1)
        top = rd.read_itp(A, validate=False)
    explicit_types = len(ttyp.extract_types(top)["angles"])
    red.average_by_type(top, categories=("bond", "angle"))
    avg_types = len(ttyp.extract_types(top)["angles"])
    assert avg_types < explicit_types, (avg_types, explicit_types)
    print(f"PASS average_by_type_collapse (angle types {explicit_types} -> {avg_types})")


def _bare_oh_alo_system():
    """A bare structure: atoms TYPED (Oh/H/Alo) with connectivity but NO params."""
    from atomipy.topology import Topology, Box, Atom, Bond, Angle
    top = Topology()
    top.box = Box.from_box_dim([3.0, 3.0, 3.0])
    top.atoms = [
        Atom(0, type="Oh", position=[0.0, 0, 0]),
        Atom(1, type="H", position=[0.097, 0, 0]),
        Atom(2, type="Alo", position=[0.19, 0.05, 0]),
    ]
    top.bonds = [Bond(0, 1)]        # Oh-H, no params
    top.angles = [Angle(1, 0, 2)]   # H-Oh-Alo, no params
    return top


def test_forcefield_itp_vs_json():
    """Parametrize the SAME bare structure from (a) GROMACS ffnonbonded.itp +
    ffbonded.itp and (b) the JSON FF library; bond / angle / LJ values must match."""
    import atomipy as ap
    ffdir = os.path.join(os.path.dirname(ap.__file__), "ffparams", "min.ff")

    ff_itp = rd.read_forcefield_itp(
        [os.path.join(ffdir, "ffnonbonded_gminff.itp"), os.path.join(ffdir, "ffbonded.itp")],
        defines=["GMINFF_k500"], include_dirs=[ffdir])
    ff_json = rd.read_forcefield_json("GMINFF/gminff_all.json", variant="GMINFF_k500")

    t_itp = _bare_oh_alo_system(); rd.apply_forcefield(t_itp, ff_itp)
    t_json = _bare_oh_alo_system(); rd.apply_forcefield(t_json, ff_json)

    # bonded params resolved (non-empty) and equal across sources
    bi, bj = t_itp.bonds[0].params, t_json.bonds[0].params
    ai, aj = t_itp.angles[0].params, t_json.angles[0].params
    assert bi and bj, (bi, bj)
    assert abs(bi["b0"] - bj["b0"]) < 1e-9 and abs(bi["k"] - bj["k"]) < 1e-6, (bi, bj)
    assert ai and aj, (ai, aj)
    assert abs(ai["theta0"] - aj["theta0"]) < 1e-6 and abs(ai["k"] - aj["k"]) < 1e-6, (ai, aj)

    # LJ resolved per atom-type and equal across sources
    lj_itp = {t.name: t.lj for t in t_itp.atom_types}
    lj_json = {t.name: t.lj for t in t_json.atom_types}
    for name in ("Oh", "Alo"):
        assert abs(lj_itp[name]["sigma"] - lj_json[name]["sigma"]) < 1e-9, (name, lj_itp[name], lj_json[name])
        assert abs(lj_itp[name]["epsilon"] - lj_json[name]["epsilon"]) < 1e-9, name
    print(f"PASS forcefield_itp_vs_json (bond b0={bi['b0']}, k={bi['k']}; "
          f"angle θ0={ai['theta0']}, k={ai['k']}; Oh σ={lj_itp['Oh']['sigma']} — .itp == .json)")


def test_xml_roundtrip():
    top = _two_oh_system()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.xml")
        wr.write_xml(top, p)
        top2 = rd.read_xml(p)
    assert top.to_dict() == top2.to_dict(), "XML round-trip not identity"
    print("PASS xml_roundtrip")


def test_schema_present_and_valid():
    from atomipy.topology import validate as v
    assert os.path.isfile(v.schema_path())
    # if jsonschema is installed, the fixture must validate clean
    issues = v.validate_against_schema(_two_oh_system())
    assert issues == [], issues
    print(f"PASS schema_present_and_valid ({'jsonschema checked' if _has_js() else 'schema file only'})")


def _has_js():
    try:
        import jsonschema  # noqa
        return True
    except Exception:
        return False


if __name__ == "__main__":
    test_units_and_prefactor()
    test_json_roundtrip()
    test_per_site_uniqueness_itp()
    test_per_site_uniqueness_data()
    test_cross_format_structural()
    test_adapter_roundtrip()
    test_typed_itp_resolution()
    test_harmonize_parity_with_legacy()
    test_average_by_type_collapse()
    test_forcefield_itp_vs_json()
    test_xml_roundtrip()
    test_schema_present_and_valid()
    print("\nALL TOPOLOGY INTERCHANGE TESTS PASSED")
