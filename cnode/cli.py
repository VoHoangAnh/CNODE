"""Command-line interface.

    cnode synth      --out data/synthetic
    cnode preprocess --config configs/ppmi.yaml --out data/ppmi_cohort
    cnode cv         --config configs/cnode.yaml --data data/ppmi_cohort --out runs/table1
    cnode sweep      --config configs/cnode.yaml --data data/ppmi_cohort --out runs/fig2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from .data.ppmi import PPMIConfig, build_cohort
from .data.synthetic import make_synthetic_cohort
from .models import MODEL_NAMES
from .train import run_cv


def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="cnode", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic PPMI-like cohort")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("preprocess", help="build the PPMI cohort from FreeSurfer outputs")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)

    s = sub.add_parser("cv", help="5-fold cross-validation of one or more models (Table I)")
    s.add_argument("--config", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--models", nargs="+", default=None, choices=MODEL_NAMES)
    s.add_argument("--epochs", type=int, default=None, help="override train.epochs")

    s = sub.add_parser("sweep", help="lambda_KL / lambda_C ratio sweep for CNODE (Fig. 2)")
    s.add_argument("--config", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.5, 1, 2, 10])
    s.add_argument("--epochs", type=int, default=None)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "synth":
        print(json.dumps(make_synthetic_cohort(a.out, seed=a.seed), indent=2))
    elif a.cmd == "preprocess":
        print(json.dumps(build_cohort(PPMIConfig(**_load_yaml(a.config)), a.out), indent=2))
    elif a.cmd == "cv":
        cfg = _load_yaml(a.config)
        if a.epochs:
            cfg["train"]["epochs"] = a.epochs
        models = a.models or cfg.get("models", ["cnode"])
        print(run_cv(a.data, a.out, models, cfg).to_string(index=False))
    elif a.cmd == "sweep":
        base = _load_yaml(a.config)
        if a.epochs:
            base["train"]["epochs"] = a.epochs
        rows = []
        for r in a.ratios:
            cfg = copy.deepcopy(base)
            # Keep lambda_KL + lambda_C constant (= 2) and vary their ratio.
            lam_c = 2.0 / (1.0 + r)
            cfg["model"]["cnode"].update(lambda_kl=r * lam_c, lambda_c=lam_c)
            cfg["eval"]["save_checkpoints"] = False
            summ = run_cv(a.data, str(Path(a.out) / f"ratio_{r:g}"), ["cnode"], cfg)
            rows.append({"ratio": r, **summ.iloc[0].drop("model").to_dict()})
        df = pd.DataFrame(rows)
        df.to_csv(Path(a.out) / "sweep.csv", index=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
