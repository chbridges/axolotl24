#!/bin/bash

set -e

[ "$#" -ge 4 ] || { echo Usage: "$0 {fi,ru} {dev,train} model" >&2; exit 1; }

python experiment1.py "$1" "$2" --full --model "$3"
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means --cosine
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means --cosine --no-pooling
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means --cosine --non-greedy
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means --cosine --no-pooling --non-greedy
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means --no-pooling
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means --no-pooling --non-greedy
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cluster-means --non-greedy
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cosine
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cosine --no-pooling
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cosine --non-greedy
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --cosine --no-pooling --non-greedy
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --no-pooling
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --no-pooling --non-greedy
python experiment1.py "$1" "$2" --full --model "$3" --clusterings 5 --non-greedy