import copy
import random
from itertools import zip_longest
import numpy as np


def create_one_bacth(examples, vocab):
    """
        examples: list of examples
        vocab: vocab
    """
    batch_size = len(examples)
    max_seq = max([len(ex[0].text) for ex in examples])
    
    batch_x = np.full((batch_size, max_seq), vocab.stoi["<pad>"], dtype=int)
    batch_y = np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        for j in range(len(examples[i][0].text)):
            batch_x[i,j] = vocab.stoi[examples[i][0].text[j]]
        if hasattr(examples[i][0], "label"):
            batch_y[i] = int(examples[i][0].label)
        else:
            batch_y[i] = 0
    return batch_x, batch_y


def create_batches(dataset, vocab, batch_size, additional_datasets=[], shuffle=True, use_elmo=True):
    """
        dataset: a torchtext.TabularDataset object
        vocab, vocab for the dataset
        batch_size: batch size 
        seed: seed to shuffle examples
        additional_datasets: a list of iterables, each iterable has the same length as the dataset
        
        return:
            a list of batches, each batch contains text data and additional data
    """
    # sanity check
    num_additional_datasets = len(additional_datasets)
    if additional_datasets:
        for d in additional_datasets:
            if d[0] is not None:
                assert len(dataset.examples) == len(d)
    
    examples = copy.deepcopy(dataset.examples)
    num_ex = len(examples)
    batches = []
    
    # zip all datasets
    examples = list(zip_longest(examples, *additional_datasets))
    
    # shuffle data and additional_datasets
    if shuffle:
        random.shuffle(examples)
    
    # divide into pools of size 100*batch_size
    pool_size = 100*batch_size
    pool_indexes = list(range(0, num_ex, pool_size)) + [num_ex]
    for s, e in zip(pool_indexes[:-1], pool_indexes[1:]):
        pool_examples = examples[s:e]
        
        # sort pool examples by sentence length
        if shuffle:
            pool_examples = sorted(pool_examples, key=lambda ex: len(ex[0].text), reverse=True)
        
        # divide into batches
        batch_indexes = list(range(0, len(pool_examples), batch_size)) + [len(pool_examples)]
        for s, e in zip(batch_indexes[:-1], batch_indexes[1:]):
            batch_examples = pool_examples[s:e]
            batch_data = create_one_bacth(batch_examples, vocab)
            # additional data
            additional_batch_data = []
            for i in range(num_additional_datasets):
                if batch_examples[0][i+1] is None:
                    additional_batch_data.append(None)
                else:
                    if i < num_additional_datasets-2:
                        additional_batch_data.append(np.array([ex[i+1] for ex in batch_examples]))
                    else:
                        if use_elmo:
                            vector_len = 1024
                            pos = -2
                        else:
                            vector_len = 768
                            pos = -1
                        seq_lens = [ex[pos].shape[0] for ex in batch_examples]
                        max_seq = max(seq_lens)
                        elmo_batch_data = np.zeros((len(batch_examples), max_seq, vector_len))
                        for i, (ex, seq_len) in enumerate(zip(batch_examples, seq_lens)):
                            elmo_batch_data[i,:seq_len,:] = ex[pos]
                        additional_batch_data.append(elmo_batch_data)
            batches.append((batch_data, additional_batch_data))
    if shuffle:        
        random.shuffle(batches)
    
    return batches
