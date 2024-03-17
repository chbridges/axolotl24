import argparse
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)

long = {"fi": "finnish", "ru": "russian"}

parser = argparse.ArgumentParser()
parser.add_argument("language", choices=["fi", "ru"], help="Language to run experiment on")
parser.add_argument(
    "split", choices=["dev", "train"], default="dev", help="Split to run experiment on"
)
parser.add_argument("pred", action="store_true", help="Run predictions")
parser.add_argument("eval", action="store_true", help="Run evaluation")
parser.add_argument(
    "--plot",
    action="store_true",
    help="Run experiments for multiple thresholds and plot the results",
)
args, positional = parser.parse_known_args()

positional = " ".join(positional)

predictor = Path("./track1.py")
scorer = Path("./axolotl24_shared_task/code/evaluation/scorer_track1.py")

gold_dir = Path(f"./axolotl24_shared_task/data/{long[args.language]}/")
gold = gold_dir / f"axolotl.{args.split}.{args.language}.tsv"
pred_dir = Path("./predictions/")
pred = pred_dir / f"pred.dev.args.{args.language}.tsv"

pred_dir.mkdir(exist_ok=True)

predict = f"python {predictor!s} --test {gold!s} --pred {pred!s} {positional}"
evaluate = f"python {scorer!s} --gold {gold!s} --pred {pred!s}"


if not args.plot:
    if args.pred:
        os.system(predict)
    if args.eval:
        os.system(evaluate)

else:
    plot_dir = Path() / "plots"
    plot_dir.mkdir(exist_ok=True)
    handle = f"{args.language}_{args.split}{''.join(positional)}"
    handle = re.sub(r"(--)|( --)| ", "_", handle)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scores = {"handle": handle, "ari": [], "f1": [], "threshold": thresholds}

    for st in thresholds:
        # Run prediction & evaluation for each threshold in range
        logging.info(f"Running experiment with threshold: {st}")
        os.system(f"{predict} --st {st}")
        os.system(evaluate)
        with (Path() / "track1_out.txt").open() as file:
            lines = [line.strip() for line in file.readlines()]
        ari = float(lines[0][5:])
        f1 = float(lines[1][4:])
        logging.info(f"Threshold: {st}\tARI: {ari}\tF1: {f1}")
        scores["ari"].append(ari)
        scores["f1"].append(f1)

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
    plt.figure()
    plt.plot(thresholds, scores["ari"], label="ARI")
    plt.plot(thresholds, scores["f1"], label="F1")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Threshold")
    plt.title(handle)
    plt.legend()
    plt.savefig(plot_dir / handle)
    logging.info(f"Saved figure to {(plot_dir / handle)!s}")
