# AXOLOTL-24: Similarity-Based Cluster Merging

Code for the paper [Similarity-Based Cluster Merging for Semantic Change Modeling](https://aclanthology.org/2024.lchange-1.3/) at [LChange'24](https://www.changeiskey.org/event/2024-acl-lchange/).

## Requirements

Run the following commands to download the [AXOLOTL-24](https://github.com/ltgoslo/axolotl24_shared_task) dataset and install dependencies:

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

Additionally, ```track1_all_parameters.py``` runs the baseline by default and comes with additional customizable parameters, not all of which have been described in the paper as they lead to uninteresting results. Run 
```
python3 track1_all_parameters.py --help
```
for more information. Not supported by ```experiment1.py``` but you can simply rename the file to fix that.

## BibTeX (ACL Anthology)

```
@inproceedings{bruckner-etal-2024-similarity,
    title = "Similarity-Based Cluster Merging for Semantic Change Modeling",
    author = {Br{\"u}ckner, Christopher  and
      Zhang, Leixin  and
      Pecina, Pavel},
    editor = "Tahmasebi, Nina  and
      Montariol, Syrielle  and
      Kutuzov, Andrey  and
      Alfter, David  and
      Periti, Francesco  and
      Cassotti, Pierluigi  and
      Huebscher, Netta",
    booktitle = "Proceedings of the 5th Workshop on Computational Approaches to Historical Language Change",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.lchange-1.3",
    pages = "23--28",
    abstract = "",
}
```
