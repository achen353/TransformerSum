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
python convert_to_extractive.py ../datasets/billsum_extractive --split_names train
python convert_to_extractive.py ../datasets/billsum_extractive --split_names validation
python convert_to_extractive.py ../datasets/billsum_extractive --split_names test --add_target_to test
python convert_to_extractive.py ../datasets/billsum_extractive --split_names ca_test --add_target_to ca_test
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
Before you train the model, make sure you've converted BillSum into an extractive summarization dataset using the 
command above.
Run the following command:
```
python main.py \
    --mode extractive \
    --data_path ../datasets/billsum_extractive \
    --weights_save_path ./trained_models \
    --do_train \
    --max_steps 100 \
    --max_seq_length 512 \
    --data_type txt \
    --by_section            # add if you're using D&C (aka DANCER) for BillSum
```
The default `--model_type` is set to be `bert`, hence the 512 for `--max_seq_length`. Modify this value depending
on your model type.

For more argument options, see the [documentation](https://transformersum.readthedocs.io/en/latest/extractive/training.html) 
for training an extractive summarizer.

```
python main.py \
    --mode extractive
    --data_path ../datasets/billsum_extractive
    --load_weights ./path/to/checkpoint.ckpt     
    --do_test     
    --max_seq_length 512 
    --by_section            # add if you're using D&C (aka DANCER) for BillSum
```

# Training an Abstractive BillSum Summarizer
Run the following command. The default dataset and preprocessing steps have been set for BillSum dataset so there's no
need to specify dataset-specific arguments.
```
python main.py \
    --mode abstractive \
    --model_name_or_path bert-base-uncased \
    --decoder_model_name_or_path bert-base-uncased \
    --do_train \
    --model_max_length 512
```
You should modify the values for `--model_name_or_path`, `--decoder_model_name_or_path`, and `--model_max_length`.
For more argument options, see the [documentation](https://transformersum.readthedocs.io/en/latest/abstractive/training.html) 
for training an abstractive summarizer.

The default value of the `--cache_file_path` option will save processed BillSum abstractive data to `../datasets/billsum_abstractive/`
```
project
└───datasets
    └───billsum_abstractive
        └───ca_test_filtered
        └───ca_test_tokenizeed
        └───test_filtered
        └───test_tokenized
        └───train_filtered
        └───train_tokenized
        └───validation_filtered
        └───validation_tokenized
    └───...
└───...
```