"""Rebuild GoBifo's repeated village indices and their paired changes from both survey waves."""

import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm

# The 2016 questions below carry new names in the Appendix's cross-wave instrument.
RENAME_2016 = dict(
    zip(
        "ms_debate ms_cons ms_demo ms_dur ms_wmn_active ms_yth_active ms_spkr ms_wmn ms_yth "
        "ms_vote chf_age ms_nochief mbr_yth".split(),
        "debate dem_cards democ duration role_wmn role_yth spkr_tot spkr_wmn spkr_yth "
        "vote rtarp_public chf_consult mbr_youth".split(),
        strict=True,
    )
)


def family_index(data, items):
    """Wave-specific control standardization, arm-mean imputation, then village item mean."""
    items = list(dict.fromkeys(items))
    values = data[items].astype(float)
    controls = data.t1.eq(0)
    centered = values - values.loc[controls].mean()
    standardized = centered / values.loc[controls].std(ddof=1).replace(0, np.nan)
    standardized.loc[controls] = standardized.loc[controls].fillna(0.0)
    treated = standardized.loc[~controls]
    standardized.loc[~controls] = treated.fillna(treated.mean())
    return standardized.mean(axis=1)


def fitted_effect(data, response):
    wards = data.filter(regex=r"^ward_\d+$")
    if not wards.sum(axis=1).eq(1).all():
        raise ValueError("Each village must belong to exactly one randomization ward")
    design = pd.concat([data[["t1", "tothhs", "road"]], wards.iloc[:, 1:]], axis=1)
    model = sm.OLS(response, sm.add_constant(design).astype(float), missing="raise").fit(
        cov_type="HC1", use_t=True
    )
    return {"effect": float(model.params["t1"]), "se": float(model.bse["t1"])}


def wave_comparison(early, late, early_items, late_items):
    """Align village identities before estimating within-village change."""
    if not early.index.is_unique or not late.index.is_unique:
        raise ValueError("Expected one observation per village per wave")
    if not early.index.sort_values().equals(late.index.sort_values()):
        raise ValueError("The two waves must contain the same villages")
    early = early.reindex(late.index)
    if not (early.t1 == late.t1).all():
        raise ValueError("Assignment changed within a village")
    first = family_index(early, early_items)
    last = family_index(late, late_items)
    return {
        "wave_2009": fitted_effect(early, first),
        "wave_2016": fitted_effect(late, last),
        "paired_change": fitted_effect(late, last - first),
        "components": [len(set(early_items)), len(set(late_items))],
    }


def sierra_replay(path):
    with ZipFile(path) as archive:
        base = "cdd-replication/"
        source = archive.read(base + "cdd-replication.do").decode("latin-1")
        macros = dict(re.findall(r'^\s*local (\w+) "([^"\r\n]*)"', source, re.MULTILINE))
        hardware_2009 = [
            item
            for i in (1, 2, 3)
            for prefix in ("p_me_corevars_", "e_me_corevars_")
            for item in macros[f"{prefix}{i}"].split()
        ]

        def names(prefix, groups):
            return [item for i in groups for item in macros[f"{prefix}{i}"].split()]

        hardware_2016 = names("vars_h", (1, 2, 3))
        common_hardware = names("vars1_h", (1, 2, 3))
        # A1 2009 family B excludes H6, H7 and H11. The later wave includes them.
        published_2009 = names("vars1_h", (4, 5, 8, 9, 10, 12))
        common_institutions = names("vars1_h", range(4, 13))
        wards = [f"ward_{i}" for i in range(1, 21)]
        design = ["id_vill", "tothhs", "road", *wards]
        early_cols = set(hardware_2009 + common_hardware + common_institutions)
        early = pd.read_stata(
            archive.open(base + "data/gbf_analysis.dta"),
            columns=design + ["post", "t", *sorted(early_cols)],
            convert_categoricals=False,
        )
        early = early.loc[early.post.eq(1)].rename(columns={"t": "t1"}).set_index("id_vill")
        reverse_names = {new: old for old, new in RENAME_2016.items()}
        late_cols = set(hardware_2016 + common_hardware)
        late_cols.update(reverse_names.get(item, item) for item in common_institutions)
        late = pd.read_stata(
            archive.open(base + "data/SES_analysis.dta"),
            columns=design + ["t1", *sorted(late_cols)],
            convert_categoricals=False,
        ).set_index("id_vill")
    late_common = late.rename(columns=RENAME_2016)
    result = {
        "communities": len(late),
        "deployments": 1,
        "hardware_table_1": wave_comparison(early, late, hardware_2009, hardware_2016),
        "hardware_common_items": wave_comparison(
            early, late_common, common_hardware, common_hardware
        ),
        "institutions_table_a1": wave_comparison(
            early, late_common, published_2009, common_institutions
        ),
        "institutions_common_items": wave_comparison(
            early, late_common, common_institutions, common_institutions
        ),
    }
    for key, effect, se in [
        ("hardware_table_1", -0.094, 0.036),
        ("hardware_common_items", -0.144, 0.037),
        ("institutions_table_a1", -0.020, 0.033),
    ]:
        observed = result[key]["paired_change"]
        np.testing.assert_allclose(
            [observed["effect"], observed["se"]], [effect, se], atol=0.00051, rtol=0
        )
    if result["communities"] != 236:
        raise ValueError("The source should contain 236 paired villages")
    return result
