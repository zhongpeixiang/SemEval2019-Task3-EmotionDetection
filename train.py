import os
import re
import io
import sys
import csv
import json
import copy
import time
import random
import pickle
import logging
import argparse
import itertools
import multiprocessing as mp
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchtext.data import Field, TabularDataset
import spacy
from spacy.symbols import ORTH

from model import RCNN
from dataloader import create_batches
from utils import label2emotion, label_weight, count_parameters, getMetrics


spacy_en = spacy.load("en")
logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")

spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


def main(config, progress):
    # save config file
    with open("./log/config_history.txt", "a+") as f:
        f.write(json.dumps(config) + "\n")
    
    logging.info("*"*80)
    logging.info("Experiment progress: {0:.2f}%".format(progress*100))
    logging.info("*"*80)

    train_all = bool(config["train_all"])
    
    # data dir
    data_dir = config["data_dir"] # data dir
    train_csv = data_dir + config["train_csv"] # train.csv or train_val.csv
    val_csv = data_dir + config["val_csv"] # val.csv or testc.csv

    # path to save model
    model_dir = config["save_dir"] # dir to save model
    f1_criteria = config["f1_criteria"] # f1 criteria to save model

    # data preprocessing settings
    min_freq = config["min_freq"] # min frequency in vocabulary
    pretrained_embedding = config["embedding_name"] # embedding name provided in torchtext
    batch_size = config["batch_size"]

    # model settings
    twitter_embedding = config["twitter_embedding"] # 0: default to word2vec or glove; 1: from datastories; 2: from trained sentiment classifier
    twitter_embedding_file = config["twitter_embedding_file"] # the saved sentiment classifier
    use_deepmoji = bool(config["use_deepmoji"])
    use_infersent = bool(config["infersent_file"])
    infersent_file = config["infersent_file"] # the infersent embedding in numpy
    use_elmo = bool(config["use_elmo"])
    use_bert_word = bool(config["use_bert_word"])
    embedding_size = config["embedding_size"]
    embedding_size = int(pretrained_embedding[-4:-1])
    if twitter_embedding > 0:
        embedding_size = 100
    
    freeze_epochs = config["freeze_epochs"] # freeze embedding for a few epochs
    kmaxpooling = config["kmaxpooling"] # top k max pooling
    hidden_size = config["hidden_size"] 
    additional_hidden_size = config["additional_hidden_size"] # an additional hidden layer before softmax
    output_size = config["output_size"] # 4-class classification
    n_layers = config["n_layers"] 
    bidirectional = bool(config["bidirectional"])
    dropout = config["dropout"]
    weight_decay = config["weight_decay"]
    recurrent_dropout = config["recurrent_dropout"]
    gradient_clip = config["gradient_clip"]

    # training settings
    num_epochs = config["epochs"]
    learning_rate = config["lr"]
    epoch_to_lower_lr = config["epoch_to_lower_lr"] # scheduled lr decay
    lr_gamma = config["lr_gamma"] # scheduled lr decay rate
    device = torch.device(config["device"]) # gpu id or "cpu"
    exp = config["exp"] # experiment number or code 
    seed = config["seed"]
    config_id = config["config_id"]

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ######################
    #### Process data ####
    ######################

    # tokenization
    logging.info("Tokenizing data {0}, {1}...".format(train_csv, val_csv))
    TEXT = Field(sequential=True, tokenize=tokenizer, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
    train_set = TabularDataset(path=train_csv, format="csv", fields=[("text", TEXT), ("label", LABEL)], skip_header=False)
    val_set = TabularDataset(path=val_csv, format="csv", fields=[("text", TEXT), ("label", LABEL)], skip_header=False)
    
    ########################
    #### Load embedding ####
    ########################
    deepmoji_train = [None]
    deepmoji_val = [None]
    if use_deepmoji:
        # load deepmoji representation
        deepmoji_file = data_dir + "deepmoji/train.npy"
        logging.info("Loading deepmoji representation from {0}".format(deepmoji_file))
        with open(deepmoji_file, "rb") as f:
            deepmoji_train = np.load(f)
        if config["val_csv"].startswith("val"):
            with open(data_dir + "deepmoji/val.npy", "rb") as f:
                deepmoji_val = np.load(f)
        elif config["val_csv"].startswith("test"):
            with open(data_dir + "deepmoji/test.npy", "rb") as f:
                deepmoji_val = np.load(f)
        if train_all:
            deepmoji_train = np.concatenate((deepmoji_train, deepmoji_val), axis=0)
    
    infersent_train = [None]
    infersent_val = [None]
    if use_infersent:
        infersent_file = data_dir + "infersent/" + infersent_file
        logging.info("Loading infersent representation from {0}".format(infersent_file))
        with open(infersent_file + "_train.npy", "rb") as f:
            infersent_train = np.load(f)
        if config["val_csv"].startswith("val"):
            with open(infersent_file + "_val.npy", "rb") as f:
                infersent_val = np.load(f)
        elif config["val_csv"].startswith("test"):
            with open(infersent_file + "_test.npy", "rb") as f:
                infersent_val = np.load(f)

    elmo_train = [None]
    elmo_val = [None]
    if use_elmo:
        elmo_file = data_dir + "elmo/"
        logging.info("Loading elmo representation from {0}".format(elmo_file))
        with open(elmo_file + "elmo_train.pkl", "rb") as f:
            elmo_train = np.load(f)
        if config["val_csv"].startswith("val"):
            with open(elmo_file + "elmo_val.pkl", "rb") as f:
                elmo_val = np.load(f)
        elif config["val_csv"].startswith("test"):
            with open(elmo_file + "elmo_test.pkl", "rb") as f:
                elmo_val = np.load(f)

    bert_word_train = [None]
    bert_word_val = [None]
    if use_bert_word:
        bert_file = data_dir + "bert/"
        logging.info("Loading bert representation from {0}".format(bert_file))
        with open(bert_file + "bert_train.pkl", "rb") as f:
            bert_word_train = np.load(f)
        if config["val_csv"].startswith("val"):
            with open(bert_file + "bert_val.pkl", "rb") as f:
                bert_word_val = np.load(f)
        elif config["val_csv"].startswith("test"):
            with open(bert_file + "bert_test.pkl", "rb") as f:
                bert_word_val = np.load(f)

    # build vocab
    logging.info("Building vocabulary...")
    if twitter_embedding == 0:
        TEXT.build_vocab(train_set, min_freq=min_freq, vectors=pretrained_embedding)
    else:
        TEXT.build_vocab(train_set, min_freq=min_freq)
    vocab_size = len(TEXT.vocab.itos)

    # use pretrained twitter embedding
    if twitter_embedding > 0:
        if twitter_embedding == 1:
            with open(data_dir + "datastories.twitter.100d.pkl", "rb") as f:
                tweet_embedding_raw = pickle.load(f)
        elif twitter_embedding == 2:
            checkpoint = torch.load("./saved_model/" + twitter_embedding_file)
            embedding = checkpoint["embedding"]
            vocab = checkpoint["vocab"]
        tweet_vectors = torch.zeros(vocab_size, embedding_size)

        if twitter_embedding != 2:
            for w, idx in TEXT.vocab.stoi.items():
                if w in tweet_embedding_raw:
                    tweet_vectors[idx] = torch.Tensor(tweet_embedding_raw[w])
                else:
                    tweet_vectors[idx] = torch.Tensor(tweet_embedding_raw["<unk>"])
        if twitter_embedding == 2:
            for w, idx in TEXT.vocab.stoi.items():
                if w in vocab.stoi:
                    tweet_vectors[idx] = torch.Tensor(embedding[vocab.stoi[w]])
                else:
                    tweet_vectors[idx] = torch.Tensor(embedding[vocab.stoi["<unk>"]])
        TEXT.vocab.vectors = tweet_vectors
    logging.info("Vocab size: {0}".format(vocab_size))

    #######################
    ### Model Training ####
    #######################
    metrics = {"accuracy" : [],
           "microPrecision" : [],
           "microRecall" : [],
           "microF1" : []}

    # create model
    logging.info("Building model...")
    model_kwargs = {
        "embed_size": embedding_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
        "dropout": dropout,
        "bidirection": bidirectional,
        "use_deepmoji": use_deepmoji,
        "use_infersent": use_infersent,
        "use_elmo": use_elmo,
        "use_bert_word": use_bert_word,
        "additional_hidden_size": additional_hidden_size,
        "recurrent_dropout": recurrent_dropout,
        "kmaxpooling": kmaxpooling,
    }
    model = globals()[config["model"]](**model_kwargs)
    logging.info("Initializing model weight...")
    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) >= 2:
            xavier_uniform_(param)
    
    if use_elmo == False:
        model.init_embedding(TEXT.vocab.vectors, config) # load GloVe 100d embedding
    logging.info(model)
    logging.info("Number of model params: {0}".format(count_parameters(model)))
    model.to(device)

    # weighted crossentropy loss
    label_weights = torch.tensor(label_weight[config["train_csv"]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=label_weights)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=epoch_to_lower_lr, gamma=lr_gamma)
    
    train_losses = []
    train_epoch_losses = []
    val_losses = []
    val_epoch_losses = []

    # train
    logging.info("Start training...")

    # freeze embedding
    model.embedding.weight.requires_grad = False

    for epoch in range(1, num_epochs + 1):

        # load data
        train_batches = create_batches(train_set, TEXT.vocab, batch_size, [deepmoji_train, infersent_train, elmo_train, bert_word_train], shuffle=True, use_elmo=use_elmo)
        val_batches = create_batches(val_set, TEXT.vocab, 1, [deepmoji_val, infersent_val, elmo_val, bert_word_val], shuffle=False, use_elmo=use_elmo)
        
        logging.info("-"*80)
        logging.critical("config_id: {0}".format(config_id))
        logging.info("Epoch {0}/{1}".format(epoch, num_epochs))

        train_epoch_loss = []
        val_epoch_loss = []

        # unfreeze embedding
        if epoch >= freeze_epochs:
            model.embedding.weight.requires_grad = True

        # lr scheduler
        scheduler.step()

        model.train()
        for batch_idx, ((batch_x, batch_y), [batch_deepmoji, batch_infersent, batch_elmo, batch_bert]) in enumerate(train_batches):
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_y = torch.from_numpy(batch_y).to(device)
            if use_deepmoji:
                batch_deepmoji = torch.from_numpy(batch_deepmoji).float().to(device)
            if use_infersent:
                batch_infersent = torch.from_numpy(batch_infersent).float().to(device)
            if use_elmo:
                batch_elmo = torch.from_numpy(batch_elmo).float().to(device)
            if use_bert_word:
                batch_bert = torch.from_numpy(batch_bert).float().to(device)

            optimizer.zero_grad()

            additional_sent_representations = {
                "deepmoji": None,
                "infersent": None,
                "elmo": None,
                "bert_word": None
            }
            if use_deepmoji:
                additional_sent_representations["deepmoji"] = batch_deepmoji
            if use_infersent:
                additional_sent_representations["infersent"] = batch_infersent
            if use_elmo:
                additional_sent_representations["elmo"] = batch_elmo
            if use_bert_word:
                additional_sent_representations["bert_word"] = batch_bert
            output = model(batch_x, config, **additional_sent_representations)
            loss = criterion(output, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # log
            train_epoch_loss.append(loss.item())
            train_losses.append(loss.item())

        logging.info("Training loss: {0:.4f}".format(np.mean(train_epoch_loss)))
        train_epoch_losses.append(np.mean(train_epoch_loss))

        # val
        if train_all == False:
            model.eval()
            eval_epoch_outputs = np.zeros((len(val_batches), output_size))
            eval_epoch_labels = np.zeros((len(val_batches), ))

            with torch.no_grad():
                for batch_idx, ((batch_x, batch_y), [batch_deepmoji, batch_infersent, batch_elmo, batch_bert]) in enumerate(val_batches):
                    batch_x = torch.from_numpy(batch_x).to(device)
                    batch_y = torch.from_numpy(batch_y).to(device)
                    if use_deepmoji:
                        batch_deepmoji = torch.from_numpy(batch_deepmoji).float().to(device)
                    if use_infersent:
                        batch_infersent = torch.from_numpy(batch_infersent).float().to(device)
                    if use_elmo:
                        batch_elmo = torch.from_numpy(batch_elmo).float().to(device)
                    if use_bert_word:
                        batch_bert = torch.from_numpy(batch_bert).float().to(device)
                    
                    additional_sent_representations = {
                        "deepmoji": None,
                        "infersent": None,
                        "elmo": None,
                        "bert_word": None
                    }
                    if use_deepmoji:
                        additional_sent_representations["deepmoji"] = batch_deepmoji
                    if use_infersent:
                        additional_sent_representations["infersent"] = batch_infersent
                    if use_elmo:
                        additional_sent_representations["elmo"] = batch_elmo
                    if use_bert_word:
                        additional_sent_representations["bert_word"] = batch_bert

                    output = model(batch_x, config, **additional_sent_representations)
                    loss = criterion(output, batch_y)

                    # log
                    val_epoch_loss.append(loss.item())
                    val_losses.append(loss.item())

                    # save predictions and labels for metrics computation
                    eval_epoch_outputs[batch_idx:batch_idx+1, :] = output.cpu().detach().numpy()
                    eval_epoch_labels[batch_idx:batch_idx+1] = batch_y.cpu().detach().numpy()

            logging.info("Validation loss: {0:.4f}".format(np.mean(val_epoch_loss)))
            val_epoch_losses.append(np.mean(val_epoch_loss))
            
            # get metrics
            logging.critical("config_id: {0}".format(config_id))
            accuracy, microPrecision, microRecall, microF1 = getMetrics(eval_epoch_outputs, eval_epoch_labels, output_size)

            # scheduler.step(microF1)

            # save model upon improvement and F1 beyond f1_criteria
            if microF1 > f1_criteria and (metrics["microF1"] == [] or microF1 > max(metrics["microF1"])):
                model_path = "{0}{1}_id_{4}_e{2}_F1_{3:.4f}.pt".format(model_dir, exp, epoch, microF1, config_id)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'model_kwargs': model_kwargs
                    }, model_path)
            metrics["accuracy"].append(accuracy)
            metrics["microPrecision"].append(microPrecision)
            metrics["microRecall"].append(microRecall)
            metrics["microF1"].append(microF1)
    
    if train_all:
        # save model
        model_path = "{0}{1}_id_{2}_e{3}.pt".format(model_dir, exp, config_id, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'model_kwargs': model_kwargs
            }, model_path)
    config.pop("seed")
    config.pop("device")
    config.pop("config_id")
    metrics["config"] = config
    return metrics

def clean_config(configs):
    cleaned_configs = []
    for config in configs:
        if config not in cleaned_configs:
            cleaned_configs.append(config)
    return cleaned_configs

def merge_metrics(metrics):
    avg_metrics = {"accuracy" : 0,
           "microPrecision" : 0,
           "microRecall" : 0,
           "microF1" : 0}
    num_metrics = len(metrics)
    for metric in metrics:
        for k in metric:
            if k != "config":
                avg_metrics[k] += np.array(metric[k])
    
    for k, v in avg_metrics.items():
        avg_metrics[k] = (v/num_metrics).tolist()

    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('--config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile) # config is now a python dict
    
    # pass experiment config to main
    # allow easy grid search for each combination of hyper-parameters
    parameters_to_search = OrderedDict() # keep keys in order
    other_parameters = {}
    for k, v in config.items():
        # if value is a list provided that key is not device, or kernel_sizes is a nested list
        if isinstance(v, list) and k != "device" and k != "kernel_sizes" and k != "epoch_to_lower_lr":
            parameters_to_search[k] = v
        elif k == "kernel_sizes" and isinstance(config["kernel_sizes"], list) and isinstance(config["kernel_sizes"][0], list):
            parameters_to_search[k] = v
        elif k == "epoch_to_lower_lr" and isinstance(config["epoch_to_lower_lr"], list) and isinstance(config["epoch_to_lower_lr"][0], list):
            parameters_to_search[k] = v
        else:
            other_parameters[k] = v

    if len(parameters_to_search) == 0:
        config_id = time.perf_counter()
        config["config_id"] = config_id
        logging.info(config)
        main(config, progress=1)
    else:
        all_configs = []
        for i, r in enumerate(itertools.product(*parameters_to_search.values())):
            specific_config = {}
            for idx, k in enumerate(parameters_to_search.keys()):
                specific_config[k] = r[idx]
            
            # merge with other parameters
            merged_config = {**other_parameters, **specific_config}

            # if a list of device is provided, distribute them evenly to these configs
            if isinstance(merged_config["device"], list):
                device = merged_config["device"][i%len(merged_config["device"])]
                merged_config["device"] = device
            all_configs.append(merged_config)
        
        # logging.info all configs
        for config in all_configs:
            config_id = time.perf_counter()
            config["config_id"] = config_id
            logging.critical("config id: {0}".format(config_id))
            logging.info(config)
            logging.info("\n")

        # multiprocessing
        num_configs = len(all_configs)
        pool = mp.Pool(processes=config["processes"])
        results = [pool.apply_async(main, args=(x,i/num_configs)) for i,x in enumerate(all_configs)]
        outputs = [p.get() for p in results]

        # if run multiple models using different seed and get the averaged result
        if "seed" in parameters_to_search:
            all_metrics = []
            all_cleaned_configs = clean_config([output["config"] for output in outputs])
            for config in all_cleaned_configs:
                metrics_per_config = []
                for output in outputs:
                    if output["config"] == config:
                        metrics_per_config.append(output)
                avg_metrics = merge_metrics(metrics_per_config)
                all_metrics.append((config, avg_metrics))
            # log metrics
            logging.info("Average evaluation result across different seeds: ")
            for config, metric in all_metrics:
                logging.info("-"*80)
                logging.info(config)
                logging.info(metric)

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for config, metric in all_metrics:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(config) + "\n")
                    f.write(json.dumps(metric) + "\n")
