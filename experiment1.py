import argparse
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

long = {"fi": "finnish", "ru": "russian"}

parser = argparse.ArgumentParser()
parser.add_argument("language", choices=["fi", "ru"], help="Language to run experiment on")
parser.add_argument(
    "split", choices=["dev", "train"], default="dev", help="Split to run experiment on"
)
parser.add_argument("--no-pred", action="store_true", help="Only evaluate")
parser.add_argument("--no-eval", action="store_true", help="Only predict")
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


if not args.plot:
    if not args.no_pred:
        os.system(f"python {predictor!s} --test {gold!s} --pred {pred!s} {positional}")
    if not args.no_eval:
        os.system(f"python {scorer!s} --gold {gold!s} --pred {pred!s}")

else:
    plot_dir = Path() / "plots"
    plot_dir.mkdir(exist_ok=True)
    handle = f"{args.language}_{args.split}{''.join(positional)}"
    handle = re.sub(r"(--)|( --)| ", "_", handle)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scores = {"ari": [], "f1": []}

    for st in thresholds:
        logging.info(f"Threshold: {st}")
        os.system(f"python {predictor!s} --test {gold!s} --pred {pred!s} --st {st} {positional}")
        os.system(f"python {scorer!s} --gold {gold!s} --pred {pred!s}")
        with (Path() / "track1_out.txt").open() as file:
            lines = [line.strip() for line in file.readlines()]
        scores["ari"].append(float(lines[0][5:]))
        scores["f1"].append(float(lines[1][4:]))

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
