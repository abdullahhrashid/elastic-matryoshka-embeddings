# Data

The dataset used for finetuning the Matryoshka Representation Learning (MRL) embedding model is based on the huge and excellent **[hard-negative-triplets](https://huggingface.co/datasets/thebajajra/hard-negative-triplets)** dataset by **thebajajra**.

This dataset contains hundreds of thousands of meticulously mined hard-negative triplets from various domains, including MS MARCO, Natural Questions (NQ), and HotpotQA. By leveraging high quality hard negatives during training, my model learns to strongly separate similar documents from subtly different but irrelevant documents, vastly improving retrieval accuracy.

## Acknowledgments

A huge thank you to **[thebajajra](https://huggingface.co/thebajajra)** for compiling, rescoring, and releasing this dataset to the open-source NLP community. Without openly accessible, high quality data like this, building efficient embedding models would be significantly harder.

If you find my work useful, please consider checking out their Hugging Face profile to see their other projects, and leave a like/star on the dataset repository to support their contributions.

## Preprocessing

The training scripts in `src/data/` stream the massive Hugging Face dataset, deduplicate the anchors, shuffle them deterministically, and cache smaller sampled subsets locally in `data/sequences` and `data/processed` to avoid downloading the entire 100GB+ dataset repeatedly during epochs.
