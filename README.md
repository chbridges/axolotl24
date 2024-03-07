# axolotl24
# AXOLOTL

# Track 1: WSI

### Data

- Finnish + Russian + **Surprise Language**
    - ⇒ We cannot fine-tune embeddings on our available languages
    - Old Russian contains **old Cyrillic** letters
- Every old word has a definition
    - New words don’t have definitions in test data
- Every new word has an example sentence
    - For old words, they are often missing or noisy in **Russian only**

### Baseline

- They compare old gloss embeddings ↔ new example embeddings
    - *Why do they use sentence embeddings for the examples? Can we make them more precise?*
- Affinity Propagation: n_clusters automated, 1 “exemplar” per cluster
- Per cluster: **Cosine similarity** to determine word sense
    - New sense if sim < threshold

### Ideas

- Extract embedding of target token in the example sentence
- Use MoverScore instead of Cosine similarity

- Different clustering algorithms/hyperparameters

# Track 2: Definition Generation

- Old words have definitions, new words don’t
