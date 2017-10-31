# Char-LSTM with Word-LSTM Relation Extraction Model

Implementation of a character-level bi-lstm relation extraction model (Kavuluru et al., 2017).

 * Uses position vectors based on offsets (off1,off2) from each entity 
 * Input form is of form [(word,off1,off2),..]
 * By default, trains and evaluates on ``example_dataset`` directory.

## Required Packages
- Python 2.7
- numpy
- tensorflow 1.0.0
- tensorflow-fold
- sklearn
- nltk

## Usage

Please note that this model takes as input the path of the data folder and not paths to each individual file. A data folder is expected to have the following files: `train_ids.txt`, `dev_ids.txt`, `test_ids.txt`, and `dataset.txt`. Please see the example_dataset directory for an example of the format. 

- More info about the dataset format can be found [here](https://github.com/AnthonyMRios/relation-extraction-rnn#data-format).

- Depending on the classes in your dataset, lines 105 and 106 in `train.py` must be changes to include them.

- To designate negative classes for the purpose of computing F1, see lines 270 in `model.py`.

### Training and Evaluating

```
python train.py --datapath=example_dataset
```

```
usage: train.py [-h] [--datapath DATAPATH] [--optimizer OPTIMIZER]
                [--batch-size BATCH_SIZE] [--num-epoch NUM_EPOCH]
                [--learning-rate LEARNING_RATE]
                [--embedding-factor EMBEDDING_FACTOR] [--decay DECAY_RATE]
                [--keep-prob KEEP_PROB] [--num-cores NUM_CORES] [--seed SEED]

Train and evaluate CharLSTM on a given dataset

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH   path to the train/dev/test dataset
  --optimizer OPTIMIZER
                        choose the optimizer: default, rmsprop, adagrad, adam.
  --batch-size BATCH_SIZE
                        number of instances in a minibatch
  --num-epoch NUM_EPOCH
                        number of passes over the training set
  --learning-rate LEARNING_RATE
                        learning rate, default depends on optimizer
  --embedding-factor EMBEDDING_FACTOR
                        learning rate multiplier for embeddings
  --decay DECAY_RATE    exponential decay for learning rate
  --keep-prob KEEP_PROB
                        dropout keep rate
  --num-cores NUM_CORES
                        seed for training
  --seed SEED           seed for training

```

## Acknowledgements

Please consider citing the following paper(s) if you use this software in your work:

> Ramakanth Kavuluru, Anthony Rios, and Tung Tran. "Extracting Drug-Drug Interactions with Word and Character-Level Recurrent Neural Networks." In Healthcare Informatics (ICHI), 2017 IEEE International Conference on, pp. 5-12. IEEE, 2017.

```
@inproceedings{kavuluru2017extracting,
  title={Extracting Drug-Drug Interactions with Word and Character-Level Recurrent Neural Networks},
  author={Kavuluru, Ramakanth and Rios, Anthony and Tran, Tung},
  booktitle={Healthcare Informatics (ICHI), 2017 IEEE International Conference on},
  pages={5--12},
  year={2017},
  organization={IEEE}
}
```

For the word-level counterpart to this model, see this [repo](https://github.com/bionlproc/relation-extraction-rnn) by Anthony Rios.
 
## Author

> Tung Tran  
> tung.tran **[at]** uky.edu  
> <http://tttran.net/>

