import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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
torch.use_deterministic_algorithms(True)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--test", help="Path to the TSV file with the test data", required=True)
    arg("--pred", help="Path to the TSV file with system predictions", required=True)
    arg("--model", help="Sentence embedding model", default="setu4993/LEALLA-large")
    arg("--st", help="Similarity threshold", type=float, default=0.3)
    arg("--no-pooling", help="Output the last hidden state without pooling", action="store_true")
    arg("--embed-targets", help="Embed only the target word in the example", action="store_true")
    arg("--cluster-means", help="Use align senses with cluster means", action="store_true")
    arg("--non-greedy", help="Align old sense in a non-greedy manner", action="store_true")
    arg("--cosine", help="Use cosine similarity as cluster affinity", action="store_true")
    arg("--pca", help="Reduce dimensionality with PCA", action="store_true")
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
def get_sentence_embeddings(outputs: ModelOutput, no_pooling: bool) -> torch.Tensor:
    if no_pooling:
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
    tokenizer, model = load_model(args)
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
            new_embeddings = get_sentence_embeddings(new_outputs, args.no_pooling)
        old_embeddings = get_sentence_embeddings(old_outputs, args.no_pooling)

        # Clustering the new representations in order to get new senses
        new_numpy = new_embeddings.detach().numpy()
        old_numpy = old_embeddings.detach().numpy()
        if args.pca:
            pca = PCA()
            new_numpy = pca.fit_transform(new_numpy)
            old_numpy = pca.transform(old_numpy)
        if args.cosine:
            ap = AffinityPropagation(random_state=42, affinity="precomputed")
            similarities = cosine_similarity(new_numpy)
            clustering = ap.fit(similarities)
        else:
            ap = AffinityPropagation(random_state=42)
            clustering = ap.fit(new_numpy)

        # Aligning the old and new senses
        if args.non_greedy:
            unique_labels = np.unique(clustering.labels_)
            similarities = np.zeros((len(unique_labels), len(senses_old)))
            for label in unique_labels:
                this_cluster = new_numpy[clustering.labels_ == label]
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
                examples_indices = np.where(clustering.labels_ == label)[0]
                examples = [new_examples[i] for i in examples_indices]
                for ex in examples:
                    exs2senses[ex] = found
        else:
            exs2senses = {}
            seen = set()
            for label in np.unique(clustering.labels_):
                found = ""
                examples_indices = np.where(clustering.labels_ == label)[0]
                examples = [new_examples[i] for i in examples_indices]
                this_cluster = new_numpy[clustering.labels_ == label]
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
