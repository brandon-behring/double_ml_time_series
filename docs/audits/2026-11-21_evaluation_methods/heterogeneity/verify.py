"""Exercise the actual Chapter 9 code, its original failure and randomized controls."""

import argparse
import ast
import contextlib
import io
import json
import re
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TEX = ROOT / "chapters/chapter_09.tex"
MDX = ROOT / "web/src/content/chapters/chapter09-heterogeneity.mdx"
SEED = 20261121


def chapter_blocks():
    tex = re.findall(
        r"\\begin\{minted\}\[[^\n]*\]\{python\}\n(.*?)\\end\{minted\}", TEX.read_text(), re.S
    )
    mdx = re.findall(r"\x60{3}python\n(.*?)\x60{3}", MDX.read_text(), re.S)
    if not tex or tex != mdx:
        raise ValueError("Chapter code blocks must match across editions")
    return tex


def projection_functions():
    block = next(b for b in chapter_blocks() if "def heldout_blp(" in b)
    tree = ast.parse(block)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"np": np, "sm": sm}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(TEX), "exec"), namespace)
    return namespace["randomized_effect_score"], namespace["heldout_blp"]


def legacy_fit(prediction):
    # Exact pre-correction Chapter 9 calculation. The method never reads a Y or T.
    return sm.OLS(prediction, sm.add_constant(prediction - prediction.mean())).fit()


def fit_summary(fit):
    return {"slope": float(fit.params[1]), "se": float(fit.bse[1]), "p": float(fit.pvalues[1])}


def calibration(repetitions):
    effect_score, project = projection_functions()
    rng = np.random.default_rng(7)
    score, error = rng.normal(size=(2, 300))
    treatment = rng.binomial(1, 0.5, len(score))
    demonstration = {
        "original": fit_summary(legacy_fit(score)),
        "corrected": fit_summary(project(score, effect_score(error, treatment, 0.5))),
    }

    rejections = {"homogeneous": [], "aligned_modifier": [], "unrelated_modifier": []}
    legacy_rejections = []
    for rep in range(repetitions):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, rep]))
        x = rng.normal(size=(600, 3))
        t = rng.binomial(1, 0.5, len(x))
        noise = rng.normal(size=len(x))
        # Same features, assignments and noise for all three scenarios. Prediction
        # depends on X0 only. X1 and X2 are independent of X0 in both samples.
        for name, tau in [
            ("homogeneous", np.ones(len(x))),
            ("aligned_modifier", 1 + 0.5 * x[:, 0]),
            ("unrelated_modifier", 1 + 0.5 * x[:, 1]),
        ]:
            y = 0.6 * x[:, 2] + t * tau + noise
            response = effect_score(y, t, 0.5)
            design = sm.add_constant(x[:, 0])
            fit = np.linalg.lstsq(design[:300], response[:300], rcond=None)[0]
            prediction = design[300:] @ fit
            heldout = project(prediction, response[300:])
            rejections[name].append(heldout.pvalues[1] < 0.05)
            if name == "homogeneous":
                legacy_rejections.append(legacy_fit(prediction).pvalues[1] < 0.05)

    def fraction(events):
        count = int(sum(events))
        proportion = count / repetitions
        return {
            "count": count,
            "fraction": proportion,
            "mcse": float(np.sqrt(proportion * (1 - proportion) / repetitions)),
        }

    return {
        "seed": SEED,
        "n_train": 300,
        "n_eval": 300,
        "repetitions": repetitions,
        "alpha": 0.05,
        "single_randomized_null_seed_7": demonstration,
        "legacy_homogeneous": fraction(legacy_rejections),
        "corrected": {name: fraction(flags) for name, flags in rejections.items()},
    }


def replay_chapter():
    import matplotlib

    matplotlib.use("Agg")
    sys.path.insert(0, str(ROOT))
    code = chapter_blocks()
    namespace = {}
    log = io.StringIO()
    with (
        tempfile.TemporaryDirectory() as folder,
        contextlib.chdir(folder),
        contextlib.redirect_stdout(log),
    ):
        for index, block in enumerate(code, start=1):
            exec(compile(block, f"{TEX}#block-{index}", "exec"), namespace)
        figures = sorted(p.name for p in Path(folder).glob("*.pdf"))
    state = namespace
    group_net = state["net_gain"]
    policy = state["assign_policy"]
    truth = state["tau_eval"]
    return {
        "versions": {p: version(p) for p in ["econml", "scikit-learn", "numpy", "statsmodels"]},
        "executed_blocks": len(code),
        "figures_generated_in_temporary_directory": figures,
        "train_people": len(state["T_train"]),
        "eval_people": len(state["T_eval"]),
        "forest_effect_rmse": float(np.sqrt(np.mean((truth - state["cate_pred"]) ** 2))),
        "linear_effect_rmse": float(np.sqrt(np.mean((truth - state["cate_linear"]) ** 2))),
        "linear_intercept": float(np.asarray(state["est_linear"].intercept_).item()),
        "average_linear_effect": float(state["cate_linear"].mean()),
        "heldout_blp": fit_summary(state["blp"]),
        "policy_treat_fraction": float(policy.mean()),
        "cost": state["cost"],
        "policy_gain": float(group_net.mean()),
        "policy_se": float(group_net.std(ddof=1) / np.sqrt(len(group_net))),
        "policy_finite_cohort_true_gain": float(np.mean(policy * (truth - state["cost"]))),
        "treat_all_gain": float((state["eval_score"] - state["cost"]).mean()),
        "gain_versus_treat_all": float(state["paired_gain"].mean()),
        "se_versus_treat_all": float(state["paired_gain"].std(ddof=1) / np.sqrt(len(truth))),
        "insurance": {
            k: dict(zip(["rmse", "predicted_sd", "true_sd"], map(float, v), strict=True))
            for k, v in state["insurance_results"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chapter",
        action="store_true",
        help="Exercise all chapter snippets using the optional full EconML environment",
    )
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    result = replay_chapter() if args.chapter else calibration(args.repetitions)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
