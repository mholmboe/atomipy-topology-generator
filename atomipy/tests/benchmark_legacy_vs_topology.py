import os, glob, tempfile, re
import atomipy as ap
import importlib; legacy = importlib.import_module("atomipy.write_top")
import atomipy.import_topology as rd
import atomipy.write_topology as wr

def angle_theta0_values(itp_path):
    vals=[]; insec=False
    for ln in open(itp_path):
        s=ln.split(';')[0].strip()
        if s.startswith('['):
            insec = re.match(r'\[\s*angles\s*\]', s, re.I) is not None
            continue
        if insec and s:
            t=s.split()
            if len(t)>=5:
                try: vals.append(round(float(t[4]),2))
                except: pass
    return vals

def data_type_counts(data_path):
    txt=open(data_path).read()
    def n(word):
        m=re.search(rf'(\d+)\s+{word}\s+types', txt)
        return int(m.group(1)) if m else 0
    return n('bond'), n('angle')

# --- real clay UC with hydroxyls (O-H bonds, M-O-H angles vary per site) ---
f=glob.glob(os.path.join(os.path.dirname(ap.__file__),'structures/minerals/UC_conf/Kaolinite*.pdb'))[0]
atoms, Box = ap.import_auto(f)
atoms, Box, _ = ap.replicate_system(atoms, Box, replicate=[2,2,1])
print(f"system: {os.path.basename(f)} x[2,2,1] -> {len(atoms)} atoms\n")

d=tempfile.mkdtemp()
A=os.path.join(d,'explicit.itp'); B=os.path.join(d,'harmonized.itp'); C=os.path.join(d,'legacy.data')

# LEGACY
legacy.itp(atoms, Box=Box, file_path=A, explicit_bonds=1, explicit_angles=1)
legacy.itp(atoms, Box=Box, file_path=B, explicit_bonds=1, explicit_angles=1,
           detect_bimodal=True, harmonize_angles=True)
try:
    legacy.lmp(atoms, Box=Box, file_path=C)
    lb, la = data_type_counts(C)
except Exception as e:
    lb=la='ERR(%s)'%str(e)[:40]

ev = angle_theta0_values(A); hv = angle_theta0_values(B)
print("LEGACY itp explicit  : %d angle rows, %d DISTINCT theta0 values" % (len(ev), len(set(ev))))
print("LEGACY itp harmonized: %d angle rows, %d DISTINCT theta0 values (cluster means)" % (len(hv), len(set(hv))))
print("LEGACY lmp .data     : %s bond types, %s angle types (type-averaged)\n" % (lb, la))

# NEW: read the explicit legacy itp -> Topology -> write_data (unique per term)
top = rd.read_itp(A, validate=False)
# carry box + positions for .data
top.box = rd.from_atoms_box(atoms, Box).box
for a, src in zip(top.atoms, atoms):
    if src.get('x') is not None:
        from atomipy.topology import units as U
        a.position=[U.angstrom_to_nm(src['x']),U.angstrom_to_nm(src['y']),U.angstrom_to_nm(src['z'])]
D=os.path.join(d,'new.data'); E=os.path.join(d,'new.itp')
wr.write_data(top, D, units='real', validate=False)
wr.write_itp(top, E, validate=False)
nb, na = data_type_counts(D)
nv = angle_theta0_values(E)
print("NEW read_itp(explicit) -> Topology: %d bonds, %d angles" % (len(top.bonds), len(top.angles)))
print("NEW write_itp explicit : %d angle rows, %d DISTINCT theta0 (should match legacy explicit)" % (len(nv), len(set(nv))))
print("NEW write_data         : %s bond types, %s angle types (per-term UNIQUE, the §16 inversion)" % (nb, na))
print()
print("round-trip explicit match:", set(ev)==set(nv))
