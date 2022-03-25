#!/usr/bin/env python
# coding: utf-8

# Import Libraries

import logging
import json
import os
import sys
import re
import unicodedata
import math
import argparse
import random
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import nltk

import torch
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq,
                          EarlyStoppingCallback,
                          set_seed,
                         )
from datasets import load_metric, Dataset


# Set Env Variables
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/", help="location of dataset")
parser.add_argument("--cache_dir", type=str, default="/data/attalk2/.cache/huggingface/transformers/", help="path location to store pre-trained models")
parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
parser.add_argument("--epochs", type=int, default=20, help="number of training iterations")
args = parser.parse_args()

# Set reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED']=str(args.seed)
set_seed(args.seed)


# Set device to GPU if available
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


## Print out progress of transformer models


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# Import Raw Dataset and Create Pandas DataFrame

translations = []
abstracts = []
pmids = []
question = []
unique = []

# Download dataset, create connected strings (with '  ' replaced by ' ') and append to lists
data = json.load(open(args.data_path + 'data.json', 'r'))

# Work through every question number
for question_number, value in data.items():
    
    # Work through every PMID
    for pmid, texts in value.items():
        if pmid != 'question':
            
            # Append abstracts and translations
            if 'translation2' in texts.keys():
                
                # If there are two translations, duplicate the abstract, pmid, and question
                for i in range(2):
                
                    abstracts.append(' '.join(texts['abstract'].values()))
                    pmids.append(pmid)
                    question.append(question_number)
                    unique.append('No')
                
                translations.append(' '.join(texts['translation1'].values()).replace('  ', ' '))
                translations.append(' '.join(texts['translation2'].values()).replace('  ', ' '))
                
            
            else:
                abstracts.append(' '.join(texts['abstract'].values()))
                pmids.append(pmid)
                question.append(question_number)
                translations.append(' '.join(texts['translation1'].values()).replace('  ', ' '))
                unique.append('Yes')
                
dataset = pd.DataFrame({'question':question, 'pmid':pmids, 'input_text':abstracts, 'target_text':translations, 'unique':unique})


## Split up dataset into train/val/test split of 70/15/15
## Rearrange train, val, and test sets so that (1) certain PMIDs are in test sets, (2) duplicate PMIDs are in the same set, (3) both train and test have abstracts from all question types

# Place certain rows in test dataframe
train, val_test = train_test_split(dataset, test_size = 0.3, random_state = args.seed, stratify = dataset[['question']])
val, test = train_test_split(val_test, test_size = 0.5, random_state = args.seed, stratify = val_test[['question']])

# Move certain rows into test dataset
special_pmids = ['25229278', '33099901', '17990843', '32941052', '34409778']

test_rows = pd.DataFrame()
val_rows = pd.DataFrame()
train_rows = pd.DataFrame()

# Reset all indices
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# Move all non-unique rows between train, test, and val
while (len(set(train.pmid).intersection(set(test.pmid))) > 0) or (len(set(val.pmid).intersection(set(test.pmid))) > 0) or (len(set(val.pmid).intersection(set(train.pmid))) > 0) or (len(train.question.value_counts()) != 50) or (len(test.question.value_counts()) != 50):
    
    # Shuffle rows if necessary
    if (test_rows.empty) and (val_rows.empty) and (train_rows.empty):
        rows = train.loc[:5, :]
        test = test.append(rows, ignore_index = True)
        train = train.drop(rows.index)
        
        rows = test.loc[:5, :]
        val = val.append(rows, ignore_index = True)
        test = test.drop(rows.index)
        
        rows = val.loc[:5, :]
        train = train.append(rows, ignore_index = True)
        val = val.drop(rows.index)
    
    # Move all special_pmid rows in val to test
    val_rows = val.loc[val['pmid'].isin(special_pmids)]
    test = test.append(val_rows, ignore_index=True)
    val = val.drop(val_rows.index)

    # Move necessary number of rows from test to val
    test_rows = test.loc[:(len(val_rows) - 1), :]
    val = val.append(test_rows, ignore_index = True)
    test = test.drop(test_rows.index)

    # Move all special_pmid rows in train to test
    train_rows = train.loc[train['pmid'].isin(special_pmids)]
    test = test.append(train_rows, ignore_index=True)
    train = train.drop(train_rows.index)

    # Move necessary number of rows from test to train
    test_rows = test.loc[:(len(train_rows) - 1), :]
    train = train.append(test_rows, ignore_index = True)
    test = test.drop(test_rows.index)
    
    
    # Move all non-unique rows between test and train to train
    test_rows = test.loc[test['pmid'].isin(set(train.pmid).intersection(set(test.pmid)))]
    train = train.append(test_rows, ignore_index=True)
    test = test.drop(test_rows.index)

    # Move necessary number of rows from train to test
    train = train.sample(frac=1).reset_index(drop=True)
    train_rows = train.loc[:(len(test_rows) - 1), :]
    test = test.append(train_rows, ignore_index = True)
    train = train.drop(train_rows.index)

    # Move all non-unique rows beteween val and test to val
    test_rows = test.loc[test['pmid'].isin(set(val.pmid).intersection(set(test.pmid)))]
    val = val.append(test_rows, ignore_index=True)
    test = test.drop(test_rows.index)

    # Move necessary number of rows from val to test
    val = val.sample(frac=1).reset_index(drop=True)
    val_rows = val.loc[:(len(test_rows) - 1), :]
    test = test.append(val_rows, ignore_index = True)
    val = val.drop(val_rows.index)
    
    # Move all non-unique rows beteween train and val to val
    train_rows = train.loc[train['pmid'].isin(set(val.pmid).intersection(set(train.pmid)))]
    val = val.append(train_rows, ignore_index=True)
    train = train.drop(train_rows.index)

    # Move necessary number of rows from val to train
    val = val.sample(frac=1).reset_index(drop=True)
    val_rows = val.loc[:(len(train_rows) - 1), :]
    train = train.append(val_rows, ignore_index = True)
    val = val.drop(val_rows.index)

    # Reset all indices
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

datasets = {'train':train, 'val':val, 'test':test}


# Train Baseline Transformer Models
def train_and_test_Transformer_Model(model_name = "t5-small", max_token_length = 512, max_token_target_length = 512, batch_size = 8, epochs = 10):
    
    # Set location to store models
    specific_model = model_name.split('/')[-1]
    checkpoint_dir = specific_model + '_runs'
    
    # Set prefix for T5 model to select summarization version of T5
    if 't5' in specific_model:
        prefix = "summarize: "
    else:
        prefix = ""

    def encode(examples):
        inputs = [prefix + doc for doc in examples["input_text"]]
        tokenized_input = tokenizer(inputs, max_length = max_token_length, truncation=True)

        with tokenizer.as_target_tokenizer():
            tokenized_label = tokenizer(examples['target_text'], max_length = max_token_target_length, truncation=True)

        tokenized_input['labels'] = tokenized_label['input_ids']
        return tokenized_input
    
    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = args.cache_dir)

    # Create & Tokenize dictionary of pandas datasets converted to Transformer Datasts
    ddatasets = {}
    for filename in list(datasets.keys()):
        ddatasets[filename] = Dataset.from_pandas(datasets[filename])
        ddatasets[filename] = ddatasets[filename].map(encode, batched=True)

    # Set model type
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir = args.cache_dir)
    
    # Set Training Arguments
    training_args = Seq2SeqTrainingArguments(
        do_train = True,
        evaluation_strategy = "epoch",
        load_best_model_at_end=True,
        logging_strategy = 'epoch',
        num_train_epochs = epochs,
        output_dir = checkpoint_dir,
        overwrite_output_dir = True,
        per_device_eval_batch_size = batch_size,
        per_device_train_batch_size = batch_size,
        predict_with_generate = True,
        remove_unused_columns=True,
        report_to="none",
        save_strategy = 'epoch',
        save_total_limit = 1,
        seed = args.seed,
    )
    
    # Set Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define Metrics
    # Convert each prediction to list type
    def prepare_preds(lst):
        return list(map(lambda el:[el], lst))
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    
    # Get ROUGE scores while training
    metric = load_metric("rouge")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    

    # Generate model predictions of test set and get the SARI, ROUGE, and BLEU scores
    def get_scores(model_type):
        decoded_preds = []
        for encoded_text in ddatasets['test']['input_ids']:
            summary_ids = model_type.generate(torch.tensor(encoded_text).unsqueeze(0),
                                         num_beams=5,
                                         no_repeat_ngram_size=2,
                                         length_penalty = 5,
                                         min_length=30,
                                         max_length=max_token_target_length,
                                         early_stopping=True)
            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            decoded_preds.append(output)
        
        decoded_labels = ddatasets['test']['target_text']
        
        # SARI just expects regular strings
        sari_decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        sari_decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # ROUGE expects newline after each sentence
        rouge_decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        rouge_decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # SACREBLEU needs some simple post-processing
        bleu_decoded_preds, bleu_decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        # SARI
        metric = load_metric('sari', seed = args.seed)
        sari_decoded_labels = prepare_preds(sari_decoded_labels)
        sari = metric.compute(sources=ddatasets['test']['input_text'], predictions = sari_decoded_preds, references = sari_decoded_labels)
        
        # ROUGE
        metric = load_metric('rouge', seed = args.seed)
        rouge = metric.compute(predictions = rouge_decoded_preds, references = rouge_decoded_labels, use_stemmer=True)
        # Extract a few results
        rouge = {key: value.mid.fmeasure * 100 for key, value in rouge.items()}
        rouge = {k: round(v, 4) for k, v in rouge.items()}
        
        # BLEU
        metric = load_metric('sacrebleu', seed = args.seed)
        bleu = metric.compute(predictions=bleu_decoded_preds, references=bleu_decoded_labels)
        bleu = {"bleu": bleu["score"]}

        bleu = {k: round(v, 4) for k, v in bleu.items()}
        
        return sari, rouge, bleu, sari_decoded_preds
    
    # Train model
    trainer = Seq2SeqTrainer(
        model, 
        training_args, 
        train_dataset=ddatasets['train'], 
        eval_dataset=ddatasets['val'],
        data_collator=data_collator,
        tokenizer = tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    
    # Save Model & Tokenizer
    trainer.save_model('models/best_run_' + specific_model)
    tokenizer.save_pretrained('models/best_run_' + specific_model)
    
    # Get Test Set Metrics
    model = AutoModelForSeq2SeqLM.from_pretrained('models/best_run_' + specific_model)
    model_sari, model_rouge, model_sacrebleu, model_preds = get_scores(model)
    print('Baseline Finetuned Model Crude SARI: ', model_sari)
    print('Baseline Finetuned Model Crude ROUGE: ', model_rouge)
    print('Baseline Finetuned Model Crude SACREBLEU: ', model_sacrebleu)
    
    # Check Results with two outputs
    print('Test Set First 2 Abstracts: ', ddatasets['test']['input_text'][0:2])
    print('Test Set First 2 PLS: ', ddatasets['test']['target_text'][0:2])
    print('\nTest Set First 2 Model Outputs: ', model_preds[0:2])
    
    return model_preds, trainer


# ## Train various models

# T5
print("T5 Train&Testing")
t5_preds, t5_trainer = train_and_test_Transformer_Model(model_name = "t5-base", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs)
datasets['test']['T5_Predictions'] = t5_preds


# Bart
print("bart Train&Testing")
bart_preds, bart_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-base", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs)
datasets['test']['BART_Predictions'] = bart_preds

# pegasus
print("pegasus Train&Testing")
pegasus_preds, pegasus_trainer = train_and_test_Transformer_Model(model_name = "google/pegasus-large", max_token_length = 512, max_token_target_length = 512, batch_size = 1, epochs = args.epochs)
datasets['test']['Pegasus_Predictions'] = pegasus_preds

# bart-large
print("bart-large Train&Testing")
bart_large_preds, bart_large_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-large-cnn", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs)
datasets['test']['BART_large_Predictions'] = bart_large_preds


# Rearrange dataframes so that second translation is in its own column
exports = {}

for key, value in datasets.items():

    questions = []
    pmids = []
    abstracts = []
    translations1 = []
    translations2 = []

    # Specific for test set
    t5_predictions = []
    bart_predictions = []
    pegasus_predictions = []
    bart_large_predictions = []

    for index, row in datasets[key].iterrows():

        # Create a new row if the pmid is unique
        if row['pmid'] not in pmids:
            questions.append(row['question'])
            pmids.append(row['pmid'])
            abstracts.append(row['input_text'])
            translations1.append(row['target_text'])
            translations2.append(np.nan)

            # Include predictions if this is the test set
            if key == 'test':
                t5_predictions.append(row['T5_Predictions'])
                bart_predictions.append(row['BART_Predictions'])
                pegasus_predictions.append(row['Pegasus_Predictions'])
                bart_large_predictions.append(row['BART_large_Predictions'])

        # Only append the second translation
        else:
            # Get the index of the first pmid
            original_pmid_index = pmids.index(row['pmid'])
            # Change second translations list
            translations2[original_pmid_index] = row['target_text']

    # Create a new dataframe to export
    exports[key] = pd.DataFrame({'question':questions, 'pmid':pmids, 'abstract':abstracts, 'translation1':translations1, 'translation2':translations2})
    if key == 'test':
        exports[key]['T5_Output'] = t5_predictions
        exports[key]['Bart_Output'] = bart_predictions
        exports[key]['Pegasus_Output'] = pegasus_predictions
        exports[key]['Bart_Large_Output'] = bart_large_predictions

# Export all the datasets 
for key, value in exports.items():
    value.to_csv(args.data_path + key + '.csv', index=False)

