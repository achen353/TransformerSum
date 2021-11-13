# Legal Document Summarization
Perform summarization (extractive and abstractive) on legal documents using machine learning transformers.

# Dataset
For our experiments, we are using the [BillSum](https://github.com/FiscalNote/BillSum) dataset. It contains abstractive
summarization of US Congressional and state bills. An example of an entry in the dataset:
```
{
    "summary": "some abstractive summary",
    "text": "some text.",
    "title": "An act to amend Section xxx."
}
```
We leverage the [BillSum](https://huggingface.co/datasets/billsum) on HuggingFace dataset for our training. Visit the 
[HuggingFace Dataset Viewer](https://huggingface.co/datasets/billsum) to examine the exact content of the dataset.
# Convert to Extractive Dataset
Use `convert_to_extractive.py` to prepare an extractive version of the BillSum dataset:
```
python convert_to_extracte.py ../datasets/billsum_extractive 
```
However, there seems to be a bug that would kill the program after 1 split. If that happens, run the above script for
each of split:
```
python convert_to_extracte.py ../datasets/billsum_extractive --split_names train
python convert_to_extracte.py ../datasets/billsum_extractive --split_names validation
python convert_to_extracte.py ../datasets/billsum_extractive --split_names test --add_target_to test
python convert_to_extracte.py ../datasets/billsum_extractive --split_names ca_test --add_target_to ca_test
```
These will create `json` files for each split:
```
project
└───datasets
    └───billsum_extractive
        └───ca_test.json
        └───test.json
        └───train.json
        └───validation.json
    └───...
└───...
```

# Training an Extractive BillSum Summarizer

# Training an Abstractive BillSum Summarizer