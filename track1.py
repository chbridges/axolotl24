import argparse
import logging
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, euclidean_distances
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.utils import ModelOutput

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
    arg("--model", help="Sentence embedding model", default="setu4993/LEALLA-large")
    arg("--st", help="Similarity threshold", type=float, default=0.3)
    arg("--clusterings", help="Number of clusterings to ensemble, 5 is fine", type=int, default=1)
    arg("--cluster-means", help="Use align senses with cluster means", action="store_true")
    arg("--cosine", help="Use cosine similarity as cluster affinity", action="store_true")
    arg("--dbscan", help="Use DBSCAN instead of Affinity Propagation", action="store_true")
    arg("--embed-targets", help="Embed only the target word in the example", action="store_true")
    arg("--ensemble-models", help="Ensemble sentence embeddings with more models", nargs="+")
    arg("--no-pooling", help="Output the last hidden state without pooling", action="store_true")
    arg("--non-greedy", help="Align old sense in a non-greedy manner", action="store_true")
    arg("--pca", help="Reduce dimensionality with PCA", action="store_true")
    arg("--sbert", help="Use sentence-transformers instead of HuggingFace", action="store_true")
    return parser.parse_args()


def load_model(
    arguments: argparse.Namespace,
) -> tuple[PreTrainedTokenizerFast, PreTrainedModel]:
    logging.info(f"Loading model {arguments.model} for sentence embeddings")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    model = AutoModel.from_pretrained(arguments.model)
    model = model.eval()
    logging.info(f"Loaded model {arguments.model}")
    return tokenizer, model


# We achieve improvements in both ARI and F1 by using the raw last hidden state of
# the CLS token instead of using pooling_output (which feeds it through BertPooler)
def get_sentence_embeddings(outputs: ModelOutput, arguments: argparse.Namespace) -> torch.Tensor:
    if arguments.no_pooling:
        return outputs.last_hidden_state[:, 0, :]
    return outputs.pooler_output


# The baseline embeds the whole example sentence, even though the sentence as a whole can have
# a different meaning from the target word. Here, we extract target word embeddings instead.
# The new embeddings have a Cosine similarity with the sentence embeddings greater than 0.9,
# but unfortunately, this approach negatively impacts the ARI and F1 scores.
def find_target_id(example: str, target: str, orth: str) -> int:
    idx = example.lower().find(orth)
    if idx == -1:
        idx = example.lower().find(target)
    return max(0, idx)


def find_token_id(inputs: BatchEncoding, batch_id: int, target_id: int) -> int:
    token_id = inputs.char_to_token(batch_id, target_id)
    if token_id is None:
        return 1
    return token_id


def get_target_embeddings(
    inputs: BatchEncoding, outputs: ModelOutput, examples: list, target: str, orths: list
) -> torch.Tensor:
    # Find the starting index of the target word in each example sentence; use 0 as a fallback
    target_ids = [find_target_id(example, target, orth) for example, orth in zip(examples, orths)]
    # Get the index of the 1st subtoken of the target word
    token_ids = [find_token_id(inputs, i, j) for i, j in enumerate(target_ids)]
    # Get the last hidden layer of this subtoken (or the baseline CLS token as a fallback)
    return torch.stack([outputs.last_hidden_state[i, j, :] for i, j in enumerate(token_ids)])


def main() -> None:
    args = parse_args()

    # sentence-transformers
    if args.sbert:
        models = [SentenceTransformer(args.model)]
        if args.ensemble_models:
            models.extend([(SentenceTransformer(model) for model in args.ensemble_models)])

    # original huggingface API
    else:
        tokenizer, model = load_model(args)
        tokenizers, models = [tokenizer], [model]

        if args.ensemble_models:
            for ensemble_model in args.ensemble_models:
                args.model = ensemble_model
                tokenizer, model = load_model(args)
                tokenizers.append(tokenizer)
                models.append(model)

    targets = pd.read_csv(args.test, sep="\t")
    for target_word in tqdm(targets.word.unique()):
        this_word = targets[targets.word == target_word]
        new = this_word[this_word[PERIOD_COLUMN] == NEW_PERIOD]
        old = this_word[this_word[PERIOD_COLUMN] == OLD_PERIOD]
        new_examples = new.example.to_list()
        new_orth = new.orth.to_list()
        new_usage_ids = new[USAGE_ID_COLUMN]
        old_glosses = [
            f"{gl} {ex}".strip() if isinstance(ex, str) else gl
            for gl, ex in zip(old.gloss.to_list(), old.example.to_list())
        ]
        senses_old = old[SENSE_ID_COLUMN].to_list()
        latin_name = senses_old[0].split("_")[0]

        # Getting representations for the new examples and old senses
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 256,
        }

        new_embeddings_per_model = []
        old_embeddings_per_model = []

        # sentence-transformers
        if args.sbert:
            for model in models:
                new_embeddings = torch.from_numpy(
                    model.encode(new_examples, show_progress_bar=False)
                )
                old_embeddings = torch.from_numpy(
                    model.encode(old_glosses, show_progress_bar=False)
                )
                new_embeddings_per_model.append(new_embeddings)
                old_embeddings_per_model.append(old_embeddings)

        # original huggingface API
        else:
            for tokenizer, model in zip(tokenizers, models):
                new_inputs = tokenizer(new_examples, **tokenizer_kwargs)
                old_inputs = tokenizer(old_glosses, **tokenizer_kwargs)
                with torch.no_grad():
                    new_outputs = model(**new_inputs)
                    old_outputs = model(**old_inputs)

                if args.embed_targets:  # Should not be used, but kept for reproducibility
                    new_embeddings = get_target_embeddings(
                        new_inputs, new_outputs, new_examples, target_word, new_orth
                    )
                else:
                    new_embeddings = get_sentence_embeddings(new_outputs, args)
                old_embeddings = get_sentence_embeddings(old_outputs, args)

                new_embeddings_per_model.append(new_embeddings)
                old_embeddings_per_model.append(old_embeddings)

        new_embeddings = torch.cat(new_embeddings_per_model, axis=1)
        old_embeddings = torch.cat(old_embeddings_per_model, axis=1)

        # Clustering the new representations in order to get new senses
        new_numpy = new_embeddings.detach().numpy()
        old_numpy = old_embeddings.detach().numpy()
        if args.pca:
            pca = PCA()
            new_numpy = pca.fit_transform(new_numpy)
            old_numpy = pca.transform(old_numpy)

        # DBSCAN
        if args.dbscan:
            if args.cosine:
                features = cosine_distances(new_numpy, new_numpy)
            else:
                features = euclidean_distances(new_numpy, new_numpy)
            eps = 2 * max(features.mean(), 0.1)
            estimators = [DBSCAN(eps=eps, min_samples=1, metric="precomputed")]

        # Affinity Propagation
        elif args.cosine:
            features = cosine_similarity(new_numpy)
            estimators = [
                AffinityPropagation(random_state=42 + i, affinity="precomputed")
                for i in range(args.clusterings)
            ]
        else:
            features = new_numpy
            estimators = [
                AffinityPropagation(random_state=42 + i) for i in range(args.clusterings)
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

        # Aligning the old and new senses
        if args.non_greedy:
            unique_labels = np.unique(clustering)
            similarities = np.zeros((len(unique_labels), len(senses_old)))
            for label in unique_labels:
                this_cluster = new_numpy[clustering == label]
                if args.cluster_means:
                    emb1 = torch.Tensor(this_cluster.mean(axis=0))
                else:
                    emb1 = torch.Tensor(this_cluster[0])
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
                examples_indices = np.where(clustering == label)[0]
                examples = [new_examples[i] for i in examples_indices]
                for ex in examples:
                    exs2senses[ex] = found
        else:
            exs2senses = {}
            seen = set()
            for label in np.unique(clustering):
                found = ""
                examples_indices = np.where(clustering == label)[0]
                examples = [new_examples[i] for i in examples_indices]
                this_cluster = new_numpy[clustering == label]
                if args.cluster_means:
                    emb1 = torch.Tensor(this_cluster.mean(axis=0))
                else:
                    emb1 = torch.Tensor(this_cluster[0])
                for old_emb, sense_old in zip(old_numpy, senses_old):
                    emb2 = torch.Tensor(old_emb)
                    if sense_old not in seen:
                        sim = F.cosine_similarity(emb1, emb2, dim=0)
                        if sim.item() >= args.st:
                            found = sense_old
                            seen.add(sense_old)
                            break
                if not found:
                    found = f"{latin_name}_novel_{label}"
                for ex in examples:
                    exs2senses[ex] = found

        assert len(new_examples) == new_usage_ids.shape[0]
        for usage_id, example in zip(new_usage_ids, new_examples):
            system_answer = exs2senses[example]
            row_number = targets[targets[USAGE_ID_COLUMN] == usage_id].index
            targets.loc[row_number, SENSE_ID_COLUMN] = system_answer
    logging.info(f"Writing the result to {args.pred}")
    targets.to_csv(args.pred, sep="\t", index=False)


if __name__ == "__main__":
    main()
