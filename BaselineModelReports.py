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
nltk.download('punkt')

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

adaptations = []
abstracts = []
pmids = []
question = []
adaptation_version = []
question_type = []
# Download dataset, create connected strings (with '  ' replaced by ' ') and append to lists
data = json.load(open(args.data_path + 'data.json', 'r'))

# Work through every question number
for question_number, value in data.items():
    
    # Work through every PMID
    for pmid, texts in value.items():
        if (pmid != 'question') and (pmid != 'question_type'):
            
            # Append abstracts and adaptations    
            # If there are multiple adaptations, duplicate the abstract, pmid, and question
            for i in range(sum('adaptation' in key for key in texts['adaptations'].keys())):
                
                pmids.append(pmid)
                question.append(question_number)
                abstracts.append(' '.join(texts['abstract'].values()))
                question_type.append(data[question_number]['question_type'])
                
            if 'adaptation1' in texts['adaptations'].keys():
                adaptations.append(' '.join(texts['adaptations']['adaptation1'].values()).replace('  ', ' '))
                adaptation_version.append(1)

                
            if 'adaptation2' in texts['adaptations'].keys():
                adaptations.append(' '.join(texts['adaptations']['adaptation2'].values()).replace('  ', ' '))
                adaptation_version.append(2)

                
            if 'adaptation3' in texts['adaptations'].keys():
                adaptations.append(' '.join(texts['adaptations']['adaptation3'].values()).replace('  ', ' '))
                adaptation_version.append(2)
        
                
dataset = pd.DataFrame({'question':question, 'pmid':pmids, 'input_text':abstracts, 'target_text':adaptations, 
                        'Adaptation_Version': adaptation_version, 'Question_Type': question_type})

## Split up dataset into train/val/test split of 70/15/15
## Rearrange train, val, and test sets so that (1) certain questions are in test/val sets and (2) annotators and question_types are stratified across all 3 datasets
need_to_shuffle = True
if ('train.csv' in os.listdir(args.data_path)) and ('val.csv' in os.listdir(args.data_path)) and ('test.csv' in os.listdir(args.data_path)):
    need_to_shuffle = False
    
if need_to_shuffle:
    test_question_numbers = ['5','12','16','22','30','36','42','48','54','61','68']
    val_question_numbers = ['2','7','13','17','26','34','40','46','52','58','66']
    train_question_numbers = [str(x) if (not str(x) in test_question_numbers) and (not str(x) in val_question_numbers) else None for x in range(1, 76)]
    train_question_numbers = [i for i in train_question_numbers if i]
    
    test = dataset.loc[dataset['question'].isin(test_question_numbers)]
    val = dataset.loc[dataset['question'].isin(val_question_numbers)]
    train = dataset.loc[dataset['question'].isin(train_question_numbers)]


if need_to_shuffle:
    datasets = {'train':train, 'val':val, 'test':test}

    # Save each to CSV file
    for key, dataset in datasets.items():
        dataset.to_csv(args.data_path + key + ".csv", index=False, encoding='utf-8-sig')
else:
    train = pd.read_csv(args.data_path + 'train.csv', header=0)
    val = pd.read_csv(args.data_path + 'val.csv', header=0)
    test = pd.read_csv(args.data_path + 'test.csv', header=0)
    datasets = {'train':train, 'val':val, 'test':test}


# Train Baseline Transformer Models
def train_and_test_Transformer_Model(model_name = "t5-small", max_token_length = 512, max_token_target_length = 512, batch_size = 8, epochs = 10, chosen_seed = 42, test_only = False):
    
    # Set location to store models
    specific_model = model_name.split('/')[-1]
    checkpoint_dir = specific_model + '_runs'
    
    # Set prefix for T5 model to select summarization version of T5
    if ('t5' in specific_model) or ('T0' in specific_model):
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create & Tokenize dictionary of pandas datasets converted to Transformer Datasts
    ddatasets = {}
    for filename in list(datasets.keys()):
        ddatasets[filename] = Dataset.from_pandas(datasets[filename])
        ddatasets[filename] = ddatasets[filename].map(encode, batched=True)

    # Set model type
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
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
        seed = chosen_seed,
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

    if not test_only:
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

# T0PP
print("T0PP Seed 7 Testing")
t0pp_7_preds_test, t0pp_trainer_test = train_and_test_Transformer_Model(model_name = "bigscience/T0_3B", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 7, test_only = True)
datasets['test']['T0PP_Test_7_Predictions'] = t0pp_7_preds_test

print("T0PP Seed 15 Testing")
t0pp_15_preds_test, t0pp_trainer_test = train_and_test_Transformer_Model(model_name = "bigscience/T0_3B", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 15, test_only = True)
datasets['test']['T0PP_Test_15_Predictions'] = t0pp_15_preds_test

print("T0PP Seed 42 Testing")
t0pp_42_preds_test, t0pp_trainer_test = train_and_test_Transformer_Model(model_name = "bigscience/T0_3B", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 42, test_only = True)
datasets['test']['T0PP_Test_42_Predictions'] = t0pp_42_preds_test

# T5
print("T5 Seed 7 Testing")
t5_7_preds_test, t5_7_trainer_test = train_and_test_Transformer_Model(model_name = "t5-base", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 7, test_only = True)
datasets['test']['T5_Test_7_Predictions'] = t5_7_preds_test

print("T5 Seed 15 Testing")
t5_15_preds_test, t5_15_trainer_test = train_and_test_Transformer_Model(model_name = "t5-base", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 15, test_only = True)
datasets['test']['T5_Test_15_Predictions'] = t5_15_preds_test

print("T5 Seed 42 Testing")
t5_42_preds_test, t5_42_trainer_test = train_and_test_Transformer_Model(model_name = "t5-base", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 42, test_only = True)
datasets['test']['T5_Test_42_Predictions'] = t5_42_preds_test

print("T5 Seed 7 Train&Testing")
t5_7_preds, t5_7_trainer = train_and_test_Transformer_Model(model_name = "t5-base", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 7)
datasets['test']['T5_7_Predictions'] = t5_7_preds

print("T5 Seed 15 Train&Testing")
t5_15_preds, t5_15_trainer = train_and_test_Transformer_Model(model_name = "t5-base", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 15)
datasets['test']['T5_15_Predictions'] = t5_15_preds

print("T5 Seed 42 Train&Testing")
t5_42_preds, t5_42_trainer = train_and_test_Transformer_Model(model_name = "t5-base", max_token_length = 512, max_token_target_length = 512, batch_size = 2, epochs = args.epochs, chosen_seed = 42)
datasets['test']['T5_42_Predictions'] = t5_42_preds

# Bart
print("bart Seed 7 Train&Testing")
bart_7_preds, bart_7_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-base", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs, chosen_seed = 7)
datasets['test']['BART_7_Predictions'] = bart_7_preds

print("bart Seed 15 Train&Testing")
bart_15_preds, bart_15_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-base", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs, chosen_seed = 15)
datasets['test']['BART_15_Predictions'] = bart_15_preds

print("bart Seed 42 Train&Testing")
bart_42_preds, bart_42_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-base", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs, chosen_seed = 42)
datasets['test']['BART_42_Predictions'] = bart_42_preds

# pegasus
print("pegasus Seed 7 Train&Testing")
pegasus_7_preds, pegasus_7_trainer = train_and_test_Transformer_Model(model_name = "google/pegasus-large", max_token_length = 512, max_token_target_length = 512, batch_size = 1, epochs = args.epochs, chosen_seed = 7)
datasets['test']['Pegasus_7_Predictions'] = pegasus_7_preds

print("pegasus Seed 15 Train&Testing")
pegasus_15_preds, pegasus_15_trainer = train_and_test_Transformer_Model(model_name = "google/pegasus-large", max_token_length = 512, max_token_target_length = 512, batch_size = 1, epochs = args.epochs, chosen_seed = 15)
datasets['test']['Pegasus_15_Predictions'] = pegasus_15_preds

print("pegasus Seed 42 Train&Testing")
pegasus_42_preds, pegasus_42_trainer = train_and_test_Transformer_Model(model_name = "google/pegasus-large", max_token_length = 512, max_token_target_length = 512, batch_size = 1, epochs = args.epochs, chosen_seed = 42)
datasets['test']['Pegasus_42_Predictions'] = pegasus_42_preds

# bart-large
print("bart-large Seed 7 Train&Testing")
bart_large_7_preds, bart_large_7_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-large-cnn", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs, chosen_seed = 7)
datasets['test']['BART_large_7_Predictions'] = bart_large_7_preds

print("bart-large Seed 15 Train&Testing")
bart_large_15_preds, bart_large_15_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-large-cnn", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs, chosen_seed = 15)
datasets['test']['BART_large_15_Predictions'] = bart_large_15_preds

print("bart-large Seed 42 Train&Testing")
bart_large_42_preds, bart_large_42_trainer = train_and_test_Transformer_Model(model_name = "facebook/bart-large-cnn", max_token_length = 1024, max_token_target_length = 1024, batch_size = 1, epochs = args.epochs, chosen_seed = 42)
datasets['test']['BART_large_42_Predictions'] = bart_large_42_preds


# Rearrange dataframes so that second adaptation is in its own column
exports = {}

for key, value in datasets.items():
    questions = []
    pmids = []
    abstracts = []
    adaptations1 = []
    adaptations2 = []

    # Specific for test set
    t0pp_7_test_predictions = []
    t0pp_15_test_predictions = []
    t0pp_42_test_predictions = []

    t5_7_test_predictions = []
    t5_15_test_predictions = []
    t5_42_test_predictions = []

    t5_7_predictions = []
    t5_15_predictions = []
    t5_42_predictions = []
    
    bart_7_predictions = []
    bart_15_predictions = []
    bart_42_predictions = []
    
    pegasus_7_predictions = []
    pegasus_15_predictions = []
    pegasus_42_predictions = []

    bart_large_7_predictions = []
    bart_large_15_predictions = []
    bart_large_42_predictions = []
    
    for index, row in datasets[key].iterrows():

        # Create a new row if the pmid is unique
        if row['pmid'] not in pmids:
            questions.append(row['question'])
            pmids.append(row['pmid'])
            abstracts.append(row['input_text'])
            adaptations1.append(row['target_text'])
            adaptations2.append(np.nan)

            # Include predictions if this is the test set
            if key == 'test':
                t0pp_7_test_predictions.append(row['T0PP_Test_7_Predictions'])
                t0pp_15_test_predictions.append(row['T0PP_Test_15_Predictions'])
                t0pp_42_test_predictions.append(row['T0PP_Test_42_Predictions'])

                t5_7_test_predictions.append(row['T5_Test_7_Predictions'])
                t5_15_test_predictions.append(row['T5_Test_15_Predictions'])
                t5_42_test_predictions.append(row['T5_Test_42_Predictions'])

                t5_7_predictions.append(row['T5_7_Predictions'])
                t5_15_predictions.append(row['T5_15_Predictions'])
                t5_42_predictions.append(row['T5_42_Predictions'])
                
                bart_7_predictions.append(row['BART_7_Predictions'])
                bart_15_predictions.append(row['BART_15_Predictions'])
                bart_42_predictions.append(row['BART_42_Predictions'])
                
                pegasus_7_predictions.append(row['Pegasus_7_Predictions'])
                pegasus_15_predictions.append(row['Pegasus_15_Predictions'])
                pegasus_42_predictions.append(row['Pegasus_42_Predictions'])

                bart_large_7_predictions.append(row['BART_large_7_Predictions'])
                bart_large_15_predictions.append(row['BART_large_15_Predictions'])
                bart_large_42_predictions.append(row['BART_large_42_Predictions'])

        # Append additional adaptations
        else: 
            # Get the index of the first pmid
            original_pmid_index = pmids.index(row['pmid'])
            
            if pd.isna(adaptations2[original_pmid_index]):
                # Change second adaptations list
                adaptations2[original_pmid_index] = row['target_text']

    # Create a new dataframe to export
    exports[key] = pd.DataFrame({'question':questions, 'pmid':pmids, 'abstract':abstracts, 'adaptation1':adaptations1, 'adaptation2':adaptations2})
    if key == 'test':
        exports[key]['T0PP_Test_7_Output'] = t0pp_7_test_predictions
        exports[key]['T0PP_Test_15_Output'] = t0pp_15_test_predictions
        exports[key]['T0PP_Test_42_Output'] = t0pp_42_test_predictions

        exports[key]['T5_Test_7_Output'] = t5_7_test_predictions
        exports[key]['T5_Test_15_Output'] = t5_15_test_predictions
        exports[key]['T5_Test_42_Output'] = t5_42_test_predictions

        exports[key]['T5_7_Output'] = t5_7_predictions
        exports[key]['T5_15_Output'] = t5_15_predictions
        exports[key]['T5_42_Output'] = t5_42_predictions
        
        exports[key]['BART_7_Output'] = bart_7_predictions
        exports[key]['BART_15_Output'] = bart_15_predictions
        exports[key]['BART_42_Output'] = bart_42_predictions
        
        exports[key]['Pegasus_7_Output'] = pegasus_7_predictions
        exports[key]['Pegasus_15_Output'] = pegasus_15_predictions
        exports[key]['Pegasus_42_Output'] = pegasus_42_predictions

        exports[key]['BART_large_7_Output'] = bart_large_7_predictions
        exports[key]['BART_large_15_Output'] = bart_large_15_predictions
        exports[key]['BART_large_42_Output'] = bart_large_42_predictions

# Export all the datasets 
for key, value in exports.items():
    value.to_csv(args.data_path + key + '_results.csv', index=False)

