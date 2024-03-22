#!/bin/bash

set -e

[ "$#" -ge 3 ] || { echo Usage: "$0 {fi,ru} {dev,train}" >&2; exit 1; }

python experiment1.py "$1" "$2" --full
python experiment1.py "$1" "$2" --full --clusterings 5 
python experiment1.py "$1" "$2" --full --clusterings 5 --cluster-means
python experiment1.py "$1" "$2" --full --clusterings 5 --cosine
python experiment1.py "$1" "$2" --full --clusterings 5 --embed-targets
python experiment1.py "$1" "$2" --full --clusterings 5 --no-pooling
python experiment1.py "$1" "$2" --full --clusterings 5 --non-greedy