"""MINFF v1.0 define names: general sets need one define, tailored sets need two."""
import warnings

import pytest

import atomipy as ap
import atomipy.gromacs as gmx
from atomipy.minff_defines import minff_defines


def test_general_set_is_one_define():
    assert minff_defines() == ["MINFF_k500"]
    assert minff_defines("MINFF_k1500") == ["MINFF_k1500"]
    assert gmx.build_defines(flexible=False) == ["-DMINFF_k500"]


def test_tailored_set_is_two_defines():
    assert minff_defines("MINFF_k500", mineral="Montmorillonite") == ["Montmorillonite", "MINFF_k500"]
    assert gmx.build_defines(mineral="Montmorillonite", ion_model="OPC3_IOD_LM", flexible=False) == [
        "-DMontmorillonite", "-DMINFF_k500", "-DOPC3_IOD_LM"]


@pytest.mark.parametrize("mineral", ["Hectorite-F", "cis_Oct_Fe2_cis", "Kaolinite"])
def test_mineral_names_with_hyphens_and_underscores(mineral):
    assert minff_defines("MINFF_k250", mineral=mineral) == [mineral, "MINFF_k250"]
    with pytest.warns(FutureWarning):
        assert minff_defines(f"{mineral}_k250") == [mineral, "MINFF_k250"]


def test_old_general_spelling_is_converted_with_a_warning():
    with pytest.warns(FutureWarning, match="GMINFF_k500"):
        assert gmx.build_defines("GMINFF_k500", flexible=False) == ["-DMINFF_k500"]


def test_old_tailored_spelling_becomes_two_defines():
    # -DMontmorillonite_k500 selects nothing any more; it must not be passed on as it is
    with pytest.warns(FutureWarning):
        assert gmx.build_defines("Montmorillonite_k500", flexible=False) == [
            "-DMontmorillonite", "-DMINFF_k500"]


def test_other_defines_and_sequences_pass_through():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert minff_defines("CLAYFF_EXT") == ["CLAYFF_EXT"]
        assert minff_defines(["MINFF_k500", "OPC3"]) == ["MINFF_k500", "OPC3"]
        assert minff_defines(None) == []


@pytest.mark.parametrize("kwargs", [
    dict(variant=None, mineral="Kaolinite"),                 # tailored needs the force constant
    dict(variant="CLAYFF_EXT", mineral="Kaolinite"),
    dict(variant="MINFF_k500", mineral="Kaolinite_k500"),     # give the mineral only
    dict(variant="Kaolinite_k500", mineral="Kaolinite"),      # mineral twice
    dict(variant="MINFF_k500", mineral="not valid"),
])
def test_tailored_set_without_both_parts_is_an_error(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(ValueError):
            minff_defines(**kwargs)


def test_bundled_json_block_keys_follow_the_same_split():
    # general keys were renamed with the defines, tailored keys were not
    general = ap.ffparams.list_blocks("GMINFF/gminff_opc3_hfe_lm_k500.json")
    assert "MINFF_k500" in general and "GMINFF_k500" not in general
    tailored = ap.ffparams.list_blocks("TMINFF/tminff_k500_all.json")
    assert "Montmorillonite_k500" in tailored and "Kaolinite_k500" in tailored
    ap.load_forcefield("GMINFF/gminff_all.json", blocks=["MINFF_k500", "OPC3_HFE_LM"])
    ap.load_forcefield("TMINFF/tminff_k500_all.json", blocks=["Montmorillonite_k500"])
