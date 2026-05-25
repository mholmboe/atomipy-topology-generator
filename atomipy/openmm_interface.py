import openmm as mm
import openmm.app as app
from openmm import unit

def _normalize_defines(defines):
    """Accept either a dict or a list of names; return a dict."""
    if isinstance(defines, dict):
        return dict(defines)
    return {name: '' for name in defines}

def load_minff_into_openmm(
    top_path,
    gro_path,
    defines,
    include_dir=None,
    nonbonded_method=None,
    nonbonded_cutoff_nm=1.0,
    constraints=None,
    rigid_water=False,
    ewald_error_tolerance=5e-4,
    use_dispersion_correction=True,
):
    """
    Build an OpenMM Topology, System, and positions from an atomipy-generated
    GROMACS topology (.top) and coordinate (.gro) pair.

    This is the recommended OpenMM entry point for MINFF simulations. The
    function is a thin wrapper around openmm.app.GromacsTopFile and
    openmm.app.GromacsGroFile that handles the standard MINFF defaults:
    flexible bonds and angles on the mineral body, OPC3-style water at the
    user's discretion, Particle Mesh Ewald for electrostatics, and the
    Lorentz-Berthelot combination rule (MINFF's convention, also OpenMM's
    default).

    Parameters
    ----------
    top_path : str
        Path to a GROMACS .top file written by atomipy (e.g. via
        ap.write_top(atoms, Box=Box, file_path='kao.top', forcefield='minff')).
        Must be self-contained: include #include directives and #ifdef blocks
        for variant selection, but resolve to a complete topology once
        preprocessed.
    gro_path : str
        Path to the matching .gro coordinate file.
    defines : dict[str, str] or list[str]
        Preprocessor variables to activate, equivalent to the GROMACS .mdp
        directive `define = -DGMINFF_k500 -DOPC3_IOD_LM -DOPC3`. Either:
          - dict form: {'GMINFF_k500': '', 'OPC3_IOD_LM': '', 'OPC3': ''}
          - list form: ['GMINFF_k500', 'OPC3_IOD_LM', 'OPC3']  (auto-converted)
        The empty-string values are sufficient for plain #ifdef checks.
    include_dir : str or None
        Directory containing the MINFF .itp files referenced by `#include`
        directives in top_path. If None, OpenMM falls back to its default
        (typically /usr/local/gromacs/share/gromacs/top), which is almost
        certainly wrong for MINFF. Provide this explicitly in production.
    nonbonded_method : openmm.app constant, optional
        Defaults to app.PME. Other valid values: app.CutoffPeriodic,
        app.CutoffNonPeriodic, app.NoCutoff, app.Ewald, app.LJPME.
    nonbonded_cutoff_nm : float
        Real-space cutoff in nanometers (default 1.0).
    constraints : openmm.app constant or None
        Default None — MINFF requires *flexible* bonds and angles on the
        mineral body for the explicit angle terms to function. Do not set
        to AllBonds or HBonds for mineral simulations.
    rigid_water : bool
        Default False (flexible water). Set True if running with a rigid
        water model (TIP3P-rigid, SPC-rigid, OPC3-rigid) — OpenMM will then
        apply SETTLE constraints to water molecules automatically.
    ewald_error_tolerance : float
        PME tolerance. Default 5e-4 matches OpenMM's standard.
    use_dispersion_correction : bool
        Long-range LJ correction. Default True matches GROMACS default.

    Returns
    -------
    topology : openmm.app.Topology
    system : openmm.System
    positions : openmm.unit.Quantity (N, 3) array in nanometers
    """
    if nonbonded_method is None:
        nonbonded_method = app.PME
    defines_dict = _normalize_defines(defines)

    gro = app.GromacsGroFile(gro_path)
    top = app.GromacsTopFile(
        top_path,
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=include_dir,
        defines=defines_dict,
    )
    system = top.createSystem(
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff_nm * unit.nanometer,
        constraints=constraints,
        rigidWater=rigid_water,
        ewaldErrorTolerance=ewald_error_tolerance,
        useDispersionCorrection=use_dispersion_correction,
    )
    return top.topology, system, gro.positions
