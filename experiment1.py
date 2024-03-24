import argparse
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def harmonic_mean(f1: float, ari: float, beta: float = 1.0) -> float:
    # Analogous to Fbeta, increasing beta increases importance of ARI
    b2 = beta**2
    return (1 + b2) * f1 * ari / (b2 * f1 + ari)


long = {"fi": "finnish", "ru": "russian"}

parser = argparse.ArgumentParser()
parser.add_argument("language", choices=["fi", "ru"], help="Language to run experiment on")
parser.add_argument(
    "split", choices=["train", "dev", "test"], default="dev", help="Split to run experiment on"
)
parser.add_argument("--pred", "-p", action="store_true", help="Run predictions")
parser.add_argument("--eval", "-e", action="store_true", help="Run evaluation")
parser.add_argument(
    "--full",
    "-f",
    action="store_true",
    help="Run experiments for multiple thresholds and store & plot the results",
)
args, positional = parser.parse_known_args()

positional = " ".join(positional)

predictor = Path("./track1.py")
scorer = Path("./axolotl24_shared_task/code/evaluation/scorer_track1.py")

gold_dir = Path(f"./axolotl24_shared_task/data/{long[args.language]}/")
gold = gold_dir / f"axolotl.{args.split}.{args.language}.tsv"
pred_dir = Path("./predictions/")
pred = pred_dir / f"pred.{args.split}.{args.language}.tsv"

pred_dir.mkdir(exist_ok=True)

predict = f"python {predictor!s} --test {gold!s} --pred {pred!s} {positional}"
evaluate = f"python {scorer!s} --gold {gold!s} --pred {pred!s}"


if not args.full:
    if args.pred:
        os.system(predict)
    if args.eval:
        os.system(evaluate)

else:
    plot_dir = Path() / "plots"
    plot_dir.mkdir(exist_ok=True)
    handle = f"{args.language}_{args.split}{''.join(positional)}"
    handle = re.sub(r"(--)|( --)|/| ", "_", handle)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    scores = {
        "handle": handle,
        "threshold": thresholds,
        "ari": [],
        "f1": [],
        "am": [],
        "hm": [],
        "hm2": [],
        "time": [],
    }

    for st in thresholds:
        # Run prediction & evaluation for each threshold in range
        logging.info(f"Running experiment with threshold: {st}")
        start = datetime.now()
        os.system(f"{predict} --st {st}")
        end = datetime.now()
        os.system(evaluate)
        with (Path() / "track1_out.txt").open() as file:
            lines = [line.strip() for line in file.readlines()]
        ari = float(lines[0][5:])
        f1 = float(lines[1][4:])
        am = 0.5 * (ari + f1)
        hm = round(harmonic_mean(f1, ari), 3)
        hm2 = round(harmonic_mean(f1, ari, 2), 3)
        logging.info(f"Threshold: {st}\tARI: {ari}\tF1: {f1}\tHM: {hm}\tHM2: {hm2}")
        scores["ari"].append(ari)
        scores["f1"].append(f1)
        scores["am"].append(am)
        scores["hm"].append(hm)
        scores["hm2"].append(hm2)
        scores["time"].append(round((end - start).total_seconds(), 3))

    # Save results
    scores_path = pred_dir / f"scores_{args.language}_{args.split}.csv"
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        scores_df = scores_df[scores_df["handle"] != handle]
        scores_df = pd.concat([scores_df, pd.DataFrame.from_dict(scores)])
    else:
        scores_df = pd.DataFrame.from_dict(scores)
    scores_df = scores_df.sort_values(by=["handle", "threshold"])
    scores_df.to_csv(scores_path, index=False)

    # Plot result and save the figure
    max_am = thresholds[np.argmax(scores["am"])]
    max_hm = thresholds[np.argmax(scores["hm"])]
    max_hm2 = thresholds[np.argmax(scores["hm2"])]
    plt.figure()
    plt.plot(thresholds, scores["ari"], color="C0", label="ARI")
    plt.plot(thresholds, scores["f1"], color="C1", label="F$_1$")
    plt.vlines(max_am, 0, 1, color="C2", linestyles="--", label="max AM")
    plt.vlines(max_hm, 0, 1, color="C3", linestyles="--", label="max HM")
    plt.vlines(max_hm2, 0, 1, color="C4", linestyles="--", label="max HM$_2$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(thresholds)
    plt.yticks(thresholds)
    plt.xlabel("Threshold")
    plt.title(handle)
    plt.legend()
    plt.savefig(plot_dir / handle)
    logging.info(f"Saved figure to {(plot_dir / handle)!s}")
