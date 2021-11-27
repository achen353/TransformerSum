# Legal Document Summarization
Perform extractive summarization on legal documents using BERT in a divide-and-conquer fasion.

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

# Results
We adopt a divide-and-conquer (D&C) methodology similar to the [DANCER](https://arxiv.org/pdf/2004.06190.pdf) approach,
conducting experiments under 5 different settings:
1. `D&C BERT (no tune)`: Directly apply `bert-base-uncased` to generate extractive summary prediction for each section of the given document and concatenate the section predictions to form the final document summary.
2. `D&C BERT (K = 1)`: This is an application of the DANCER approach (main D&C approach). Before training, break down the ground truth summary by sentence. For each summary sentence, find the sentence in the original text having the large ROUGE-1 and ROUGE-2 scores with the summary sentence. Assign the summary sentence to the corresponding section.
3. `D&C BERT (K = 2)`: Similar to `D&C BERT (K = 1)` except that each summary sentence is assign to 2 sections.
4. `D&C BERT (K = 3)`: Similar to `D&C BERT (K = 1)` except that each summary sentence is assign to 3 sections.
5. `D&C BERT (no sec)`: This is a simplification of the D&C approach. For each section in the original text, we just correspond it to the ground truth summary of the entire document.

### Congressional Bills (`test` split) 
|                            | Rouge-1 F-1 | Rouge-2 F-1 | Rouge-L F-1 |
| -------------------------- |------:|------:|------:|
| SumBasic                   | 30.56 | 15.33 | 23.75 |
| LSA                        | 32.24 | 14.02 | 23.75 |
| TextRank                   | 34.10 | 17.45 | 27.57 |
| DOC                        | 38.18 | 21.22 | 31.02 |
| SUM                        | 41.29 | 24.47 | 34.07 |
| DOC + SUM                  | 41.28 | 24.31 | 34.15 |
| PEAGUS (BASE)              | 51.42 | 29.68 | 37.78 |
| PEAGUS (LARGE - C4)        | 57.20 | 39.56 | 45.80 |
| PEAGUS (LARGE - HugeNews)  | 57.31 | 40.19 | 45.82 |
| OURS: D&C BERT (no tune)   | 44.45 | 24.04 | 41.37 |
| OURS: D&C BERT (K = 1)     | 45.10 | 24.26 | 41.26 |
| OURS: D&C BERT (K = 2)     | 53.70 | 35.26 | 51.44 |
| OURS: D&C BERT (K = 3)     | 51.99 | 33.47 | 49.69 |
| OURS: D&C BERT (no sec)    | 53.33 | 35.36 | 51.19 |

### CA Bills (`ca_test` split) 
|                            | Rouge-1 F-1 | Rouge-2 F-1 | Rouge-L F-1 |
| -------------------------- |------:|------:|------:|
| SumBasic                   | 35.47 | 16.16 | 30.10 |
| LSA                        | 35.05 | 16.34 | 30.10 |
| TextRank                   | 35.81 | 18.10 | 30.10 |
| DOC                        | 37.32 | 18.72 | 31.87 |
| SUM                        | 38.67 | 20.59 | 33.11 |
| DOC + SUM                  | 39.25 | 21.16 | 33.77 |
| PEAGUS (BASE)              | n/a | n/a | n/a |
| PEAGUS (LARGE - C4)        | n/a | n/a | n/a |
| PEAGUS (LARGE - HugeNews)  | n/a | n/a | n/a |
| OURS: D&C BERT (no tune)   | 51.70 | 42.30 | 51.16 |
| OURS: D&C BERT (K = 1)     | 33.54 | 22.12 | 30.82 |
| OURS: D&C BERT (K = 2)     | 50.89 | 42.76 | 50.89 |
| OURS: D&C BERT (K = 3)     | 50.89 | 42.76 | 50.89 |
| OURS: D&C BERT (no sec)    | 50.89 | 42.76 | 50.89 |

## Training Configs
All the training is done with the default hyper-parameters for this program (details available soon).

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
# Testing an Extractive BillSum Summarizer
Use the `--do_test` flag instead of `do_train` and enable `--by_section` for calculating the D&C performance on BillSum.

The project contains two different ROUGE score calculations: `rouge-score` and `pyrouge`. `rouge-score` is the default 
option. It is a pure python implementation of ROUGE designed to replicate the results of the official ROUGE package. 
While this option is cleaner (no perl installation required, no temporary directories, faster processing) than using 
`pyrouge`, this option should not be used for official results due to minor score differences with `pyrouge`.

You will need to perform extra installation steps for `pyrouge`. Refer to this [post](https://stackoverflow.com/a/57686103/11526586)
for the steps.
```
# Add `--by-section` if you're using D&C (aka DANCER) for BillSum

python main.py \
    --mode extractive \
    --data_path ../datasets/billsum_extractive \
    --load_weights ./path/to/checkpoint.ckpt \
    --do_test \
    --max_seq_length 512 \
    --by_section \
    --test_use_pyrouge      # we want official ROUGE score results
```

# Training an Abstractive BillSum Summarizer
We only conduct experiments on extractive summarization. However, this repo is also capable of training abstractive
summarizer.

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