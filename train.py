#!/usr/bin/env python
import sys, os, random, pickle, json
import numpy as np
from model import CharacterLSTM
import sklearn.metrics as skm
import argparse

parser = argparse.ArgumentParser(description='Train and evaluate CharLSTM on a given dataset')
parser.add_argument('--datapath', dest='datapath', type=str,
                    default='example_dataset', 
                    help='path to the train/dev/test dataset')
parser.add_argument('--optimizer', dest='optimizer', type=str,
                    default='default', 
                    help='choose the optimizer: default, rmsprop, adagrad, adam.')
parser.add_argument('--batch-size', dest='batch_size', type=int, 
                    default=24, help='number of instances in a minibatch')
parser.add_argument('--num-epoch', dest='num_epoch', type=int, 
                    default=30, help='number of passes over the training set')
parser.add_argument('--learning-rate', dest='learning_rate', type=str,
                    default='default', help='learning rate, default depends on optimizer')
parser.add_argument('--embedding-factor', dest='embedding_factor', type=float,
                    default=1.0, help='learning rate multiplier for embeddings')
parser.add_argument('--decay', dest='decay_rate', type=float,
                    default=1.0, help='exponential decay for learning rate')
parser.add_argument('--keep-prob', dest='keep_prob', type=float,
                    default=0.8, help='dropout keep rate')
parser.add_argument('--num-cores', dest='num_cores', type=int, 
                    default=4, help='seed for training')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=1, help='seed for training')

def main(args):
    print "Running CharLSTM model"
    print args
    random.seed(args.seed)
    
    print "Loading dataset.."
    trainset, devset, testset = load_dataset(args.datapath)
    
    print "Loaded {}/{}/{} instances from train/dev/test set".format(len(trainset), len(devset), len(testset))
    
    X_train, y_train = zip(*trainset) 
    X_dev, y_dev = zip(*devset)
    X_test, y_test = zip(*testset)
    
    labels = sorted(list(set(y_train + y_test)))
    
    # Create the model, passing in relevant parameters
    model = CharacterLSTM(labels=labels,
                        optimizer=args.optimizer,
                        embedding_size=250, 
                        lstm_dim=200, 
                        num_cores=args.num_cores,
                        embedding_factor=args.embedding_factor,
                        learning_rate=args.learning_rate,
                        decay_rate=args.decay_rate,
                        dropout_keep=args.keep_prob)
    
    model_path = './scratch/saved_model_d{}_s{}'.format(hash(args.datapath),args.seed)
    if not os.path.exists(model_path + '.meta'):
        if not os.path.exists('./scratch'):
            os.mkdir('./scratch')
            
        print "Training.."
        model.fit(X_train, y_train, X_dev, y_dev, 
                num_epoch=args.num_epoch,
                batch_size=args.batch_size,
                seed=args.seed)
        
        model.save(model_path)
    else:
        model.restore(model_path)
    
    print "Evaluating.."
    micro_evaluation = model.evaluate(X_test,y_test,macro=False)
    macro_evaluation = model.evaluate(X_test,y_test,macro=True)
    print "Micro Test Eval: F={:.4f} P={:.4f} R={:.4f}".format(*micro_evaluation)
    print "Macro Test Eval: F={:.4f} P={:.4f} R={:.4f}".format(*macro_evaluation)

def load(fname):
    data = []
    with open(fname,'r') as f:
        for l in f:
             data.append(json.loads(l))
    
    return data

def load_dataset(datapath):
    examples = []
    with open(datapath + '/dataset.txt','r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            if line.strip() == '':
                continue
            
            a = line.strip()
            b = f.readline().strip()
            doc_id, sent_id, e1, e2, label = b.split('\t')
            
            example = []
            tokens = a.split(' ')
            e1 = tokens.index('DRUGA')
            e2 = tokens.index('DRUGB')
                
            for i in range(len(tokens)):
                example.append((tokens[i],i-e1,i-e2))
            
            examples.append((doc_id, example,label))
    
    train_ids = []
    with open(datapath + '/train_ids.txt','r') as f:
        for l in f:
            if l.strip() == '':
                continue
            
            train_ids.append(l.strip())
    
    dev_ids = []
    with open(datapath + '/dev_ids.txt','r') as f:
        for l in f:
            if l.strip() == '':
                continue
            
            dev_ids.append(l.strip())
    
    test_ids = []
    with open(datapath + '/test_ids.txt','r') as f:
        for l in f:
            if l.strip() == '':
                continue
            
            test_ids.append(l.strip())
    
    trainset = []
    devset = []
    testset = []
    for instance in examples:
        doc_id, example, label = instance
        if doc_id in train_ids:
            trainset.append((example,label))
        elif doc_id in dev_ids:
            devset.append((example,label))
        elif doc_id in test_ids:
            testset.append((example,label))
    
    return trainset, devset, testset

if __name__ == '__main__':
    main(parser.parse_args())
