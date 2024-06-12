import argparse
import logging
import random
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

NEW_PERIOD = "new"
OLD_PERIOD = "old"
SENSE_ID_COLUMN = "sense_id"
USAGE_ID_COLUMN = "usage_id"
PERIOD_COLUMN = "period"

torch.manual_seed(0)
random.seed(0)
np.random.default_rng(0)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--test", help="Path to the TSV file with the test data", required=True)
    arg("--pred", help="Path to the TSV file with system predictions", required=True)
    arg("--st", help="Similarity threshold", type=float, default=0.3)
    arg("--clusterings", help="Number of clusterings to ensemble, 5 is fine", type=int, default=5)
    arg(
        "--model",
        help="Sentence embedding model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    arg(
        "--ensemble-models",
        help="Ensemble sentence embeddings with more models",
        nargs="+",
        default=["sentence-transformers/LaBSE"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = [SentenceTransformer(args.model)]
    if args.ensemble_models:
        models.extend([SentenceTransformer(model) for model in args.ensemble_models])

    targets = pd.read_csv(args.test, sep="\t")
    for target_word in tqdm(targets.word.unique()):
        this_word = targets[targets.word == target_word]
        new = this_word[this_word[PERIOD_COLUMN] == NEW_PERIOD]
        old = this_word[this_word[PERIOD_COLUMN] == OLD_PERIOD]
        new_examples = new.example.to_list()
        new_usage_ids = new[USAGE_ID_COLUMN]
        old_glosses = [
            f"{gl} {ex}".strip() if isinstance(ex, str) else gl
            for gl, ex in zip(old.gloss.to_list(), old.example.to_list())
        ]
        senses_old = old[SENSE_ID_COLUMN].to_list()
        latin_name = senses_old[0].split("_")[0]

        # Getting representations for the new examples and old senses
        new_embeddings_per_model = []
        old_embeddings_per_model = []

        for model in models:
            new_embeddings = torch.from_numpy(model.encode(new_examples, show_progress_bar=False))
            old_embeddings = torch.from_numpy(model.encode(old_glosses, show_progress_bar=False))
            new_embeddings_per_model.append(new_embeddings)
            old_embeddings_per_model.append(old_embeddings)

        new_embeddings = torch.cat(new_embeddings_per_model, axis=1)
        old_embeddings = torch.cat(old_embeddings_per_model, axis=1)

        # Clustering the new representations in order to get new senses
        new_numpy = new_embeddings.detach().numpy()
        old_numpy = old_embeddings.detach().numpy()

        features = cosine_similarity(new_numpy)
        estimators = [
            AffinityPropagation(random_state=42 + i, affinity="precomputed")
            for i in range(args.clusterings)
        ]

        ensemble_clusterings = np.vstack([est.fit_predict(features) for est in estimators])
        ensemble_clusterings[ensemble_clusterings == -1] = 0

        clustering = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=ensemble_clusterings,
        )

        ## avoid errors from voting (unique cluster_label [0,2]) with the code below
        cluster_unique = np.unique(clustering)
        ## detect the missing labels in label sequence
        fill = [i for i in range(len(cluster_unique)) if i not in cluster_unique]
        ## change the labels
        for i in fill:
            clustering_idx = np.where(clustering == cluster_unique.max())[0]
            cluster_unique_idx = np.where(cluster_unique == cluster_unique.max())[0]
            clustering[clustering_idx] = i
            cluster_unique[cluster_unique_idx] = i

        # Align the old and new senses
        novel_labels = []
        novel_senses = []

        unique_labels = np.unique(clustering)
        similarities = np.zeros((len(unique_labels), len(senses_old)))
        for label in unique_labels:
            this_cluster = new_numpy[clustering == label]
            emb1 = torch.Tensor(this_cluster.mean(axis=0))
            for old_sense, old_emb in enumerate(old_numpy):
                emb2 = torch.Tensor(old_emb)
                sim = F.cosine_similarity(emb1, emb2, dim=0)
                similarities[label, old_sense] = sim
        # assign old senses to labels where sim > threshold
        exs2senses = {}
        closest_senses = similarities.argmax(axis=1)
        for label, old_sense in zip(unique_labels, closest_senses):
            if similarities[label, old_sense] >= args.st:
                found = senses_old[old_sense]
            else:
                found = f"{latin_name}_novel_{label}"
                novel_labels.append(label)
                novel_senses.append(found)
            examples_indices = np.where(clustering == label)[0]
            examples = [new_examples[i] for i in examples_indices]
            for ex in examples:
                exs2senses[ex] = found

        # 2nd pass to merge novel sense clusters
        n = len(novel_labels)
        if n:
            similarities = np.zeros((n, n))

            for i in range(n - 1):
                label_i = novel_labels[i]
                cluster_i = new_numpy[clustering == label_i]
                emb_i = torch.Tensor(cluster_i.mean(axis=0))
                for j in range(i + 1, n):
                    label_j = novel_labels[j]
                    cluster_j = new_numpy[clustering == label_j]
                    emb_j = torch.Tensor(cluster_j.mean(axis=0))
                    sim = F.cosine_similarity(emb_i, emb_j, dim=0)
                    similarities[i, j] = similarities[j, i] = sim

            closest_senses = similarities.argmax(axis=1)
            to_merge = []
            for i, j in enumerate(closest_senses):
                if similarities[i, j] >= args.st:
                    to_merge.append((min(i, j), max(i, j)))  # tuples so the pairs can be hashed
            to_merge = [list(x) for x in set(to_merge)]  # lists so they can me merged
            for x, y in combinations(to_merge, 2):
                if not set(x).isdisjoint(y):
                    x += y
                    if y in to_merge:
                        to_merge.remove(y)
            to_merge = [sorted(set(x)) for x in to_merge]  # remove duplicates from lists

            # overwrite senses to merge
            for labels in to_merge:
                sense = novel_senses[labels[0]]
                for i in labels[1:]:
                    examples_indices = np.where(clustering == novel_labels[i])[0]
                    examples = [new_examples[j] for j in examples_indices]
                    for ex in examples:
                        exs2senses[ex] = sense

        assert len(new_examples) == new_usage_ids.shape[0]
        for usage_id, example in zip(new_usage_ids, new_examples):
            system_answer = exs2senses[example]
            row_number = targets[targets[USAGE_ID_COLUMN] == usage_id].index
            targets.loc[row_number, SENSE_ID_COLUMN] = system_answer
    logging.info(f"Writing the result to {args.pred}")
    targets.to_csv(args.pred, sep="\t", index=False)


if __name__ == "__main__":
    main()
