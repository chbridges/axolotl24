#!/bin/bash

set -e

[ "$#" -ge 3 ] || { echo Usage: "$0 {fi,ru} {dev,train}" >&2; exit 1; }

python experiment1.py "$1" "$2" --full
python experiment1.py "$1" "$2" --full --cluster-means
python experiment1.py "$1" "$2" --full --cluster-means --cosine
python experiment1.py "$1" "$2" --full --cluster-means --no-pooling
python experiment1.py "$1" "$2" --full --cluster-means --non-greedy
python experiment1.py "$1" "$2" --full --cluster-means --cosine --no-pooling
python experiment1.py "$1" "$2" --full --cluster-means --cosine --non-greedy
python experiment1.py "$1" "$2" --full --cluster-means --cosine --no-pooling --non-greedy
python experiment1.py "$1" "$2" --full --cosine
python experiment1.py "$1" "$2" --full --cosine --no-pooling
python experiment1.py "$1" "$2" --full --cosine --non-greedy
python experiment1.py "$1" "$2" --full --cosine --no-pooling --non-greedy
python experiment1.py "$1" "$2" --full --no-pooling
python experiment1.py "$1" "$2" --full --no-pooling --non-greedy
python experiment1.py "$1" "$2" --full --non-greedy
