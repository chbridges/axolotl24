import argparse
import os
from pathlib import Path

long = {"fi": "finnish", "ru": "russian"}

parser = argparse.ArgumentParser()
parser.add_argument("language", choices=["fi", "ru"], help="Language to run experiment on")
parser.add_argument(
    "split", choices=["dev", "train"], default="dev", help="Split to run experiment on"
)
parser.add_argument("--no-pred", action="store_true", help="Only evaluate")
parser.add_argument("--no-eval", action="store_true", help="Only predict")
args, positional = parser.parse_known_args()

positional = " ".join(positional)

predictor = Path("./track1.py")
scorer = Path("./axolotl24_shared_task/code/evaluation/scorer_track1.py")

gold_dir = Path(f"./axolotl24_shared_task/data/{long[args.language]}/")
gold = gold_dir / f"axolotl.{args.split}.{args.language}.tsv"
pred_dir = Path("./predictions/")
pred = pred_dir / f"pred.dev.args.{args.language}.tsv"

pred_dir.mkdir(exist_ok=True)

if not args.no_pred:
    os.system(f"python {predictor!s} --test {gold!s} --pred {pred!s} {positional}")
if not args.no_eval:
    os.system(f"python {scorer!s} --gold {gold!s} --pred {pred!s}")
