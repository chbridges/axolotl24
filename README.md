# AXOLOTL-24: Similarity-Based Cluster Merging

Code for the paper "Similarity-Based Cluster Merging for Semantic Change Modeling" at [LChange'24](https://www.changeiskey.org/event/2024-acl-lchange/).

## Requirements

Download the AXOLOTL-24 dataset and install dependencies by running the following commands:

```
git submodule init  
pip3 install -r requirements.txt
```

## Usage

The code for our best model ```track1.py``` can be run the same way as the [baseline code](https://github.com/ltgoslo/axolotl24_shared_task/tree/main/code/baselines).

We recommend using the higher-level wrapper ```experiment1.py``` to automate the predict-evaluate workflow, e.g., for the Finnish dev set at threshold 0.2:
```
python3 experiment1.py fi dev --pred --eval --st 0.2
```

Additionally, ```track1_all_parameters.py``` runs the baseline by default and comes with additional customizable parameters, not all of which have been described in the paper as they lead to uninteresting results. Run ```python3 track1_all_parameters.py --help``` for more information. Not supported by ```experiment1.py``` but you can simply rename the file to fix that.

## Reference

