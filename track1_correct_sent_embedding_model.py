import argparse
import logging
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation,DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,cosine_distances
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--test", help="Path to the TSV file with the test data", required=True)
    arg("--pred", help="Path to the TSV file with system predictions", required=True)
    arg("--model", help="Sentence embedding model", default="setu4993/LEALLA-large")
    arg("--st", help="Similarity threshold", type=float, default=0.3)
    arg("--cluster-algorithm", help = "name of cluster_algorithm", choices=['ap','DBSCAN'],default='ap')
    arg("--clusterings", help="Number of clusterings to ensemble, 5 is fine", type=int, default=1)
    arg("--cluster-means", help="Use align senses with cluster means", action="store_true")
    arg("--cosine", help="Use cosine similarity as cluster affinity", action="store_true")
    arg("--embed-targets", help="Embed only the target word in the example", action="store_true")
    arg("--ensemble-models", help="Ensemble sentence embeddings with more models", nargs="+")
    arg("--no-pooling", help="Output the last hidden state without pooling", action="store_true")
    arg("--non-greedy", help="Align old sense in a non-greedy manner", action="store_true")
    arg("--pca", help="Reduce dimensionality with PCA", action="store_true")
    return parser.parse_args()


def load_model(
    arguments: argparse.Namespace,
):
    logging.info(f"Loading model {arguments.model} for sentence embeddings")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    model = AutoModel.from_pretrained(arguments.model)
    model = model.eval()
    logging.info(f"Loaded model {arguments.model}")
    return tokenizer, model

def get_embeddings(sentences,tokenizer,model):
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 256,
    }
    sent_inputs = tokenizer(sentences, **tokenizer_kwargs)
    with torch.no_grad():
        sent_outputs = model(**sent_inputs).pooler_output 
    return sent_outputs


def get_embeddings_from_sentence_transformer(loaded_model,sentences):
    # sent_model = SentenceTransformer(model_name)
    embeddings = loaded_model.encode(sentences)
    return torch.from_numpy(embeddings)

def main():
    args = parse_args()

    all_models = [args.model]
    if args.ensemble_models:
        all_models += args.ensemble_models
    word_models = [i for i in all_models if 'sentence-transformers/' not in i]
    sent_models = [i for i in all_models if 'sentence-transformers/' in i]
    loaded_sent_models = [SentenceTransformer(i) for i in sent_models]
    tokenizers, loaded_word_models = [],[]
    if len(word_models)!= 0:
        for model in word_models:
            tokenizer, model = load_model(args)
            tokenizers.append(tokenizer)
            loaded_word_models.append(model)

        

    targets = pd.read_csv(args.test, sep="\t")
    for target_word in tqdm(targets.word.unique()):
        this_word = targets[targets.word == target_word]
        new = this_word[this_word[PERIOD_COLUMN] == NEW_PERIOD]
        old = this_word[this_word[PERIOD_COLUMN] == OLD_PERIOD]
        new_examples = new.example.to_list()
        # new_orth = new.orth.to_list()
        new_usage_ids = new[USAGE_ID_COLUMN]
        old_glosses = [
            f"{gl} {ex}".strip() if isinstance(ex, str) else gl
            for gl, ex in zip(old.gloss.to_list(), old.example.to_list())
        ]
        senses_old = old[SENSE_ID_COLUMN].to_list()
        latin_name = senses_old[0].split("_")[0]


        new_embeddings_per_model = []
        old_embeddings_per_model = []

        # Getting representations for the new examples and old senses
        if len(tokenizers)!=0:
            for tokenizer, model in zip(tokenizers, loaded_word_models):
                new_outputs = get_embeddings(new_examples,tokenizer,model)
                old_outputs = get_embeddings(old_glosses,tokenizer,model)
                new_embeddings_per_model.append(new_outputs)
                old_embeddings_per_model.append(old_outputs)

        if len(loaded_sent_models)!=0:
            for model in loaded_sent_models:
                new_outputs = get_embeddings_from_sentence_transformer(model, new_examples)
                old_outputs = get_embeddings_from_sentence_transformer(model, old_glosses)       
                new_embeddings_per_model.append(new_outputs)
                old_embeddings_per_model.append(old_outputs)
        
        new_embeddings = torch.cat(new_embeddings_per_model, axis=1)
        old_embeddings = torch.cat(old_embeddings_per_model, axis=1)

        # Clustering the new representations in order to get new senses
        new_numpy = new_embeddings.detach().numpy()
        old_numpy = old_embeddings.detach().numpy()
        if args.cluster_algorithm =='ap':
            if args.pca:
                pca = PCA()
                new_numpy = pca.fit_transform(new_numpy)
                old_numpy = pca.transform(old_numpy)

            if args.cosine:
                features = cosine_similarity(new_numpy)
                estimators = [
                    AffinityPropagation(random_state=42+i, affinity="precomputed")
                    for i in range(args.clusterings)
                ]
            else:
                features = new_numpy
                estimators = [AffinityPropagation(random_state=42+i) for i in range(args.clusterings)]

            ensemble_clusterings = np.vstack([ap.fit_predict(features) for ap in estimators])
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
                clustering_idx = np.where(clustering==cluster_unique.max())[0]
                cluster_unique_idx = np.where(cluster_unique==cluster_unique.max())[0]
                clustering[clustering_idx] = i
                cluster_unique[cluster_unique_idx] = i

    
        elif args.cluster_algorithm =='DBSCAN':
            if args.cosine:
                matrix = cosine_distances(new_numpy,new_numpy)
                mean_value = matrix.mean()
            else:
                matrix = euclidean_distances(new_numpy,new_numpy)
                mean_value = matrix.mean()
            if mean_value == 0: ## (0 will get error, so smoothing)
                mean_value = 0.1
            clustering = DBSCAN(eps=2*mean_value, min_samples=1, metric='precomputed').fit(matrix).labels_

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
