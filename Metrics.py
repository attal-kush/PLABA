#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

from __future__ import division
import logging
import json
import os
import sys
import re
import unicodedata
import math
import random
from typing import Optional
from collections import Counter
import math
import itertools

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import textstat
from scipy.stats import kendalltau, wilcoxon

import torch
from transformers import set_seed
from datasets import load_metric, Dataset


# ## Define D-SARI Method

## Recreate D-SARI Metric
def D_SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)
    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()

    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()

    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0

    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]

        # print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]

    keepscore_precision = 0

    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)

    keepscore_recall = 0

    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)

    keepscore = 0

    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    # DELETION

    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter

    deltmpscore1 = 0
    deltmpscore2 = 0

    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]

    delscore_precision = 0

    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    delscore_recall = 0

    if len(delgramcounterall_rep) > 0:
        delscore_recall = deltmpscore1 / len(delgramcounterall_rep)

    delscore = 0

    if delscore_precision > 0 or delscore_recall > 0:
        delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)

    # ADDITION

    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)
    addtmpscore = 0

    for addgram in addgramcountergood:
        addtmpscore += 1

    addscore_precision = 0
    addscore_recall = 0

    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)

    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)

    addscore = 0

    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)

def count_length(ssent, csent, rsents):

    input_length = len(ssent.split(" "))
    output_length = len(csent.split(" "))
    reference_length = 0

    for rsent in rsents:
        reference_length += len(rsent.split(" "))

    reference_length = int(reference_length / len(rsents))

    return input_length, reference_length, output_length

def sentence_number(csent, rsents):

    output_sentence_number = len(nltk.sent_tokenize(csent))
    reference_sentence_number = 0

    for rsent in rsents:
        reference_sentence_number += len(nltk.sent_tokenize(rsent))

    reference_sentence_number = int(reference_sentence_number / len(rsents))

    return reference_sentence_number, output_sentence_number

def D_SARIsent(ssent, csent, rsents):
    numref = len(rsents)

    s1grams = ssent.lower().split(" ")
    c1grams = csent.lower().split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []
    
    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []

    for rsent in rsents:

        r1grams = rsent.lower().split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)

        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)

            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)

            if i < len(r1grams) - 3:
                r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                r4grams.append(r4gram)

        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    for i in range(0, len(s1grams) - 1):

        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)

        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)

        if i < len(s1grams) - 3:
            s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
            s4grams.append(s4gram)

    for i in range(0, len(c1grams) - 1):

        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i + 1]
            c2grams.append(c2gram)

        if i < len(c1grams) - 2:
            c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
            c3grams.append(c3gram)

        if i < len(c1grams) - 3:
            c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
            c4grams.append(c4gram)

    (keep1score, del1score, add1score) = D_SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = D_SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = D_SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = D_SARIngram(s4grams, c4grams, r4gramslist, numref)
    
    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    input_length, reference_length, output_length = count_length(ssent, csent, rsents)
    reference_sentence_number, output_sentence_number = sentence_number(csent, rsents)
    
    if output_length >= reference_length:
        LP_1 = 1
    else:
        LP_1 = math.exp((output_length - reference_length) / output_length)

    if output_length > reference_length:
        LP_2 = math.exp((reference_length - output_length) / max(input_length - reference_length, 1))
    else:
        LP_2 = 1

    SLP = math.exp(-abs(reference_sentence_number - output_sentence_number) / max(reference_sentence_number,
                                                                                  output_sentence_number))
    avgkeepscore = avgkeepscore * LP_2 * SLP
    avgaddscore = avgaddscore * LP_1
    avgdelscore = avgdelscore * LP_2
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3

    return finalscore


# # Set Env Variables

DATA_PATH = "data/"
SEED = 42

# Set reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
set_seed(SEED)


# Set device to GPU if available
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ## Print out progress of transformer models

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# # Import Datasets

keys = ['train', 'val', 'test']
datasets = {}
for key in keys:
    datasets[key] = pd.read_csv(DATA_PATH + key + '_results.csv', header = 0)
data = pd.concat(datasets.values(), ignore_index = True).sort_values(by='question', ascending=True).reset_index(drop=True)
text_columns = list(data.columns[2:])
model_names = ['T5', 'Bart', 'Pegasus', 'Bart_Large']

print('Number of abstracts:', len(data))
print('Number of adaptations:', len(pd.concat([data['adaptation1'], data['adaptation2']], axis = 0).dropna()))
print('Number of model outputs:', len(datasets['test']))


# ## Calculate word count and sentence count for questions

full_json = pd.read_json(DATA_PATH + 'data.json')
# Get pandas series of word length of questions
question_words = []
question_sents = []
for key, value in full_json.items():
    question_words.append(int(len(re.findall(r'\w+', value['question']))))
    question_sents.append(int(len(nltk.sent_tokenize(value['question']))))
    
question_words_pd = pd.Series(question_words)
question_sents_pd = pd.Series(question_sents)

print("Average word count for questions:", round(question_words_pd.mean()))
print("Average standard deviation for questions:", round(question_words_pd.std()))

print("Average sentence count for questions:", round(question_sents_pd.mean()))
print("Average standard deviation for questions:", round(question_sents_pd.std()))


# ## Calculate word count and sentence count

for column in text_columns:
    # Sentence count
    data[column + '_sentencecount'] = data[column].apply(lambda x: int(len(nltk.sent_tokenize(x))) if isinstance(x, str) else np.nan)
    # Word count
    data[column + '_wordcount'] = data[column].apply(lambda x: int(len(re.findall(r'\w+', x))) if isinstance(x, str) else np.nan)


text_amounts = ['wordcount', 'sentencecount']

for text_stat in text_amounts:
    print('\n' + text_stat.upper() + '\n')
    
    # Abstract
    print('Abstracts\n')
    print("Average " + text_stat + " for Abstracts:", round(data['abstract_'+text_stat].mean()))
    print("Average standard deviation for Abstracts:", round(data['abstract_'+text_stat].std()))

    # Adaptations
    print('\nAdaptations\n')
    adaptation_series = pd.concat([data['adaptation1_'+text_stat], data['adaptation2_'+text_stat]], axis = 0)
    print("Average " + text_stat + " for adaptations:", round(adaptation_series.mean()))
    print("Average standard deviation for adaptations:", round(adaptation_series.std()))

    # Model Outputs
    print('\nModel Outputs\n')
    for model in model_names:
        print("Average " + text_stat + " for " + model + ":", round(data[model + '_Output_'+text_stat].mean()))
        print("Average standard deviation for " + model + ":", round(data[model + '_Output_'+text_stat].std()))


# ## Show some automatically generated examples

for index, row in datasets['test'].iterrows():
    if (row['question'] == 5) | (row['question'] == 12):    
        print('Question:', row['question'])
        print('T5 Output:', row['T5_Output'])
        print('PEGASUS Output:', row['Pegasus_Output'])
        print('BART Output:', row['Bart_Output'])
        print('BART-Large-CNN Output:', row['Bart_Large_Output'])
    


# ## Calculate Inter-Annotator BLEU and ROUGE

trans1_annotators = [-1, 4, 5, -1, 6, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 3, 3, 3, 3, 2, 2, 2, 3, 1, 1, 3, 2, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

map_annotators_by_question = {}
for question_number in range(1, 76):
    map_annotators_by_question[question_number] = trans1_annotators[question_number - 1]
   
# Add to dataframe
data['trans1_annotator'] = data['question'].map(map_annotators_by_question)


rouge_measures = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    
        
for annotator in set(trans1_annotators):
    if annotator > -1:
        print('Annotator:', annotator)
        common_adaptation_indices = set(data['adaptation2'].dropna().index.tolist()) & set(data['trans1_annotator'][data['trans1_annotator'] == annotator].index.tolist())

        # ROUGE
        rouge_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in data['adaptation1'][common_adaptation_indices]]
        rouge_refs = [["\n".join(nltk.sent_tokenize(pred.strip()))] for pred in data['adaptation2'][common_adaptation_indices]]
        metric = load_metric('rouge', seed = SEED)
        rouge = metric.compute(predictions = rouge_preds, references = rouge_refs, use_stemmer=True)
        # Extract a few results
        rouge = {key: value.mid.fmeasure * 100 for key, value in rouge.items()}
        print("Inter-Annotator ROUGE Metrics:", rouge)

        # Add ROUGE metrics to dataset
        data['Annotator' + str(annotator) + '_Agreement_ROUGE'] = data.apply(lambda x: metric.compute(predictions = ["\n".join(nltk.sent_tokenize(x['adaptation1']))], references = [["\n".join(nltk.sent_tokenize(x['adaptation2']))]], use_stemmer=True) if (pd.notna(x['adaptation1']) and pd.notna(x['adaptation2']) and (x.name in common_adaptation_indices)) else np.nan, axis=1)
        for rouge_stat in rouge_measures:
            data['Annotator' + str(annotator) + 'Agreement_'+rouge_stat.upper()] = data['Annotator' + str(annotator) + '_Agreement_ROUGE'].apply(lambda x: (x[rouge_stat].mid.fmeasure * 100) if pd.notna(x) else np.nan)

        # SACREBLEU needs some simple post-processing
        bleu_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in data['adaptation1'][common_adaptation_indices]]
        bleu_refs = [[" ".join(nltk.sent_tokenize(pred.strip()))] for pred in data['adaptation2'][common_adaptation_indices]]
        metric = load_metric('sacrebleu', seed = SEED)
        bleu = metric.compute(predictions=bleu_preds, references=bleu_refs)
        print('Inter-Annotator BLEU Metrics:', bleu['score'])

        # Add SACREBLEU metrics to dataset
        data['Annotator' + str(annotator) + '_Agreement_BLEU'] = data.apply(lambda x: metric.compute(predictions = [" ".join(nltk.sent_tokenize(x['adaptation1']))], references = [[" ".join(nltk.sent_tokenize(x['adaptation2']))]])['score'] if (pd.notna(x['adaptation1']) and pd.notna(x['adaptation2']) and (x.name in common_adaptation_indices)) else np.nan, axis=1)
    


# ## Check for low BLEU between annotators

for annotator in set(trans1_annotators):
    if annotator > -1:
        print('Annotator:', annotator)
        min_id = data['Annotator' + str(annotator) + '_Agreement_BLEU'].idxmin()
        print('BLEU Score:', data['Annotator' + str(annotator) + '_Agreement_BLEU'].min())
        print('Question:', data.loc[min_id, 'question'])
        print('adaptation1:', data.loc[min_id, 'adaptation1'])
        print('adaptation2:', data.loc[min_id, 'adaptation2'])


# ## Calculate Flesch-Kincaid for (1) source text, (2) adaptations, (3) all predictions

for column in text_columns:
    data[column + '_FKGL'] = data[column].apply(lambda x: textstat.flesch_kincaid_grade(x) if isinstance(x, str) else np.nan)
    # Same for test set
    datasets['test'][column + '_FKGL'] = datasets['test'][column].apply(lambda x: textstat.flesch_kincaid_grade(x) if isinstance(x, str) else np.nan)


# ## Use Kendall-Tau to Check Pairwise Comparisions

# Find average FKGL levels
print('FKGL Abstract Avg:', round(data['abstract_FKGL'].mean(), 2))
print('FKGL Abstract St Dev:', round(data['abstract_FKGL'].std(), 2))

# Abstract vs all adaptations
adaptation_series = data[['adaptation1_FKGL', 'adaptation2_FKGL']].mean(axis=1)
print('FKGL adaptations Avg:', round(adaptation_series.mean(), 2))
print('FKGL adaptations St Dev:', round(adaptation_series.std(), 2))
corr, p = kendalltau(data['abstract_FKGL'], adaptation_series)
print('FKGL Abstract vs adaptations Correlation:', round(corr, 2))
print('FKGL Abstract vs adaptations P-Value:', p)
print('\n')

# Within test set
for model in model_names:
    print(model)
    print('FKGL '+model+' Avg:', round(data[model+'_Output_FKGL'].mean(), 2))
    print('FKGL '+model+' St Dev:', round(data[model+'_Output_FKGL'].std(), 2))
    # Abstract vs Model
    corr, p = kendalltau(datasets['test']['abstract_FKGL'], datasets['test'][model+'_Output_FKGL'])
    print('FKGL Abstract vs '+model+' Correlation:', round(corr, 2))
    print('FKGL Abstract vs '+model+' P-Value:', p)
    print('\n')
    # Model vs adaptations
    adaptation_series = datasets['test'][['adaptation1_FKGL', 'adaptation2_FKGL']].mean(axis=1)
    corr, p = kendalltau(datasets['test'][model+'_Output_FKGL'], adaptation_series)
    print('FKGL '+model+' vs adaptations Correlation:', round(corr, 2))
    print('FKGL '+model+' vs adaptations P-Value:', p)
    print('\n')


# ## Calculate overall (1) D-SARI, (2) SARI, (3) ROUGE, (4) BLEU for all model outputs

# ## D-Sari and SARI

for model in model_names:
    ## Model
    # D-SARI
    datasets['test'][model + '_D-SARI'] = datasets['test'][['abstract', 'adaptation1', 'adaptation2', model+'_Output']].apply(lambda x: D_SARIsent(x['abstract'], x[model+'_Output'], [x['adaptation1'], x['adaptation2']]) if pd.notna(x['adaptation2']) else D_SARIsent(x['abstract'], x[model+'_Output'], [x['adaptation1']]), axis=1)
    print(model+' D-SARI Average:', datasets['test'][model+'_D-SARI'].mean())

    # SARI
    sari_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in datasets['test'][model+'_Output']]
    sari_refs = [[" ".join(nltk.sent_tokenize(trans1.strip())), " ".join(nltk.sent_tokenize(trans2.strip()))] if pd.notna(trans2) else [" ".join(nltk.sent_tokenize(trans1.strip()))] for trans1, trans2 in zip(datasets['test']['adaptation1'], datasets['test']['adaptation2'])]
    sari_sources = datasets['test']['abstract'].tolist()
    metric = load_metric('sari', seed = SEED)
    sari = metric.compute(sources=sari_sources, predictions = sari_preds, references = sari_refs)
    print(sari)

# ## ROUGE and BLEU

# ## Model Output vs adaptations

for model in model_names:
    ## Model
    print(model)

    # ROUGE
    rouge_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in datasets['test'][model+'_Output']]
    rouge_refs = [["\n".join(nltk.sent_tokenize(trans1.strip())), "\n".join(nltk.sent_tokenize(trans2.strip()))] if pd.notna(trans2) else ["\n".join(nltk.sent_tokenize(trans1.strip()))] for trans1, trans2 in zip(datasets['test']['adaptation1'], datasets['test']['adaptation2'])]
    metric = load_metric('rouge', seed = SEED)
    rouge = metric.compute(predictions = rouge_preds, references = rouge_refs, use_stemmer=True)
    # Extract a few results
    rouge = {key: value.mid.fmeasure * 100 for key, value in rouge.items()}
    print(rouge)

    # SACREBLEU needs some simple post-processing
    bleu_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in datasets['test'][model+'_Output']]
    bleu_refs = [[" ".join(nltk.sent_tokenize(trans1.strip())), " ".join(nltk.sent_tokenize(trans2.strip()))] if pd.notna(trans2) else [" ".join(nltk.sent_tokenize(trans1.strip())), " ".join(nltk.sent_tokenize(trans1.strip()))] for trans1, trans2 in zip(datasets['test']['adaptation1'], datasets['test']['adaptation2'])]
    metric = load_metric('sacrebleu', seed = SEED)
    bleu = metric.compute(predictions=bleu_preds, references=bleu_refs)
    print('BLEU:', bleu['score'])
    print("\n")


# ## Model Output vs Abstracts

for model in model_names:
    print(model)
    # ROUGE
    rouge_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in datasets['test'][model+'_Output']]
    rouge_sources = [["\n".join(nltk.sent_tokenize(source.strip()))] for source in datasets['test']['abstract']]
    metric = load_metric('rouge', seed = SEED)
    rouge = metric.compute(predictions = rouge_preds, references = rouge_sources, use_stemmer=True)
    # Extract a few results
    rouge = {key: value.mid.fmeasure * 100 for key, value in rouge.items()}
    print(rouge)

    # SACREBLEU needs some simple post-processing
    bleu_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in datasets['test'][model+'_Output']]
    bleu_sources = [[" ".join(nltk.sent_tokenize(source.strip()))] for source in datasets['test']['abstract']]
    metric = load_metric('sacrebleu', seed = SEED)
    bleu = metric.compute(predictions=bleu_preds, references=bleu_sources)
    print('BLEU:', bleu['score'])
    print("\n")


# ## Abstracts vs adaptations

# ROUGE
rouge_refs = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in data['abstract']]
rouge_sources = [["\n".join(nltk.sent_tokenize(trans1.strip())), "\n".join(nltk.sent_tokenize(trans2.strip()))] if pd.notna(trans2) else ["\n".join(nltk.sent_tokenize(trans1.strip()))] for trans1, trans2 in zip(data['adaptation1'], data['adaptation2'])]
metric = load_metric('rouge', seed = SEED)
rouge = metric.compute(predictions = rouge_sources, references = rouge_refs, use_stemmer=True)
# Extract a few results
rouge = {key: value.mid.fmeasure * 100 for key, value in rouge.items()}
print(rouge)

# SACREBLEU needs some simple post-processing
bleu_refs = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in data['abstract']]
bleu_sources = [[" ".join(nltk.sent_tokenize(trans1.strip())), " ".join(nltk.sent_tokenize(trans2.strip()))] if pd.notna(trans2) else [" ".join(nltk.sent_tokenize(trans1.strip())), " ".join(nltk.sent_tokenize(trans1.strip()))] for trans1, trans2 in zip(data['adaptation1'], data['adaptation2'])]
metric = load_metric('sacrebleu', seed = SEED)
bleu = metric.compute(predictions=bleu_sources, references=bleu_refs)
print('BLEU:', bleu['score'])


# ## Calculate individual (1) D-SARI, (2) SARI, (3) ROUGE, (4) BLEU for all model outputs and then perform Wilcoxon signed-rank test

# ## D-Sari and SARI

for model in model_names:
    metric = load_metric('sari', seed = SEED)
    datasets['test'][model+'_SARI'] = datasets['test'].apply(lambda x: metric.compute(sources=[" ".join(nltk.sent_tokenize(x['abstract'].strip()))], predictions=[x[model+'_Output']], references = [[" ".join(nltk.sent_tokenize(x['adaptation1'].strip())), " ".join(nltk.sent_tokenize(x['adaptation2'].strip()))]])['sari'] if pd.notna(x['adaptation2']) else metric.compute(sources=[" ".join(nltk.sent_tokenize(x['abstract'].strip()))], predictions=[x[model+'_Output']], references = [[" ".join(nltk.sent_tokenize(x['adaptation1'].strip()))]])['sari'], axis = 1)


# ## Wilcoxon-Ranked Sum Test

model_pairs = itertools.combinations(model_names, 2)
for pair in model_pairs:
    ## D-SARI
    print(pair[0] + ' D-SARI avg:', datasets['test'][pair[0]+'_D-SARI'].mean())
    print(pair[1] + ' D-SARI avg:', datasets['test'][pair[1]+'_D-SARI'].mean())
    print(wilcoxon(datasets['test'][pair[0]+'_D-SARI'], datasets['test'][pair[1]+'_D-SARI']))
    
    ## SARI
    print(pair[0] + ' SARI avg:', datasets['test'][pair[0]+'_SARI'].mean())
    print(pair[1] + ' SARI avg:', datasets['test'][pair[1]+'_SARI'].mean())
    print(wilcoxon(datasets['test'][pair[0]+'_SARI'], datasets['test'][pair[1]+'_SARI']))
    print('\n')


# ## ROUGE and BLEU

# ## Model Output vs adaptations

# ROUGE
metric = load_metric('rouge', seed = SEED)
for model in model_names:
    datasets['test'][model+'vsadaptations_ROUGE'] = datasets['test'].apply(lambda x: metric.compute(predictions=["\n".join(nltk.sent_tokenize(x[model+'_Output'].strip()))], references = [["\n".join(nltk.sent_tokenize(x['adaptation1'].strip())), "\n".join(nltk.sent_tokenize(x['adaptation2'].strip()))]], use_stemmer=True) if pd.notna(x['adaptation2']) else metric.compute(predictions=["\n".join(nltk.sent_tokenize(x[model+'_Output'].strip()))], references = [["\n".join(nltk.sent_tokenize(x['adaptation1'].strip()))]], use_stemmer=True), axis = 1)


rouge_measures = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
for model in model_names:
    for rouge_stat in rouge_measures:
        datasets['test'][model+'vsadaptations_'+rouge_stat.upper()] = datasets['test'][model+'vsadaptations_ROUGE'].apply(lambda x: x[rouge_stat].mid.fmeasure * 100)

# BLEU
metric = load_metric('sacrebleu', seed = SEED)
for model in model_names:
    datasets['test'][model+'vsadaptations_BLEU'] = datasets['test'].apply(lambda x: metric.compute(predictions=[" ".join(nltk.sent_tokenize(x[model+'_Output'].strip()))], references = [[" ".join(nltk.sent_tokenize(x['adaptation1'].strip())), " ".join(nltk.sent_tokenize(x['adaptation2'].strip()))]])['score'] if pd.notna(x['adaptation2']) else metric.compute(predictions=[" ".join(nltk.sent_tokenize(x[model+'_Output'].strip()))], references = [[" ".join(nltk.sent_tokenize(x['adaptation1'].strip())), " ".join(nltk.sent_tokenize(x['adaptation1'].strip()))]])['score'], axis = 1)


model_pairs = itertools.combinations(model_names, 2)
for pair in model_pairs:
    ## ROUGE
    for rouge_stat in rouge_measures:
        print(pair[0] + ' vs adaptations ' + rouge_stat.upper() + ' avg:', datasets['test'][pair[0]+'vsadaptations_'+rouge_stat.upper()].mean())
        print(pair[1] + ' vs adaptations ' + rouge_stat.upper() + ' avg:', datasets['test'][pair[1]+'vsadaptations_'+rouge_stat.upper()].mean())
        print(wilcoxon(datasets['test'][pair[0]+'vsadaptations_'+rouge_stat.upper()], datasets['test'][pair[1]+'vsadaptations_'+rouge_stat.upper()]))
    
    ## BLEU
    print(pair[0] + ' vs adaptations BLEU avg:', datasets['test'][pair[0]+'vsadaptations_BLEU'].mean())
    print(pair[1] + ' vs adaptations BLEU avg:', datasets['test'][pair[1]+'vsadaptations_BLEU'].mean())
    print(wilcoxon(datasets['test'][pair[0]+'vsadaptations_BLEU'], datasets['test'][pair[1]+'vsadaptations_BLEU']))
    print('\n')


# ## Model Output vs Abstracts

# ROUGE
metric = load_metric('rouge', seed = SEED)
for model in model_names:
    datasets['test'][model+'vsAbstract_ROUGE'] = datasets['test'].apply(lambda x: metric.compute(predictions=["\n".join(nltk.sent_tokenize(x[model+'_Output'].strip()))], references = [["\n".join(nltk.sent_tokenize(x['abstract'].strip()))]], use_stemmer=True), axis = 1)


for model in model_names:
    for rouge_stat in rouge_measures:
        datasets['test'][model+'vsAbstract_'+rouge_stat.upper()] = datasets['test'][model+'vsAbstract_ROUGE'].apply(lambda x: x[rouge_stat].mid.fmeasure * 100)


# BLEU
metric = load_metric('sacrebleu', seed = SEED)
for model in model_names:
    datasets['test'][model+'vsAbstract_BLEU'] = datasets['test'].apply(lambda x: metric.compute(predictions=[" ".join(nltk.sent_tokenize(x[model+'_Output'].strip()))], references = [[" ".join(nltk.sent_tokenize(x['adaptation1'].strip()))]])['score'], axis = 1)


model_pairs = itertools.combinations(model_names, 2)
for pair in model_pairs:
    ## ROUGE
    for rouge_stat in rouge_measures:
        print(pair[0] + ' vs Abstract ' + rouge_stat.upper() + ' avg:', datasets['test'][pair[0]+'vsAbstract_'+rouge_stat.upper()].mean())
        print(pair[1] + ' vs Abstract ' + rouge_stat.upper() + ' avg:', datasets['test'][pair[1]+'vsAbstract_'+rouge_stat.upper()].mean())
        print(wilcoxon(datasets['test'][pair[0]+'vsAbstract_'+rouge_stat.upper()], datasets['test'][pair[1]+'vsAbstract_'+rouge_stat.upper()]))
    
    ## BLEU
    print(pair[0] + ' vs Abstract BLEU avg:', datasets['test'][pair[0]+'vsAbstract_BLEU'].mean())
    print(pair[1] + ' vs Abstract BLEU avg:', datasets['test'][pair[1]+'vsAbstract_BLEU'].mean())
    print(wilcoxon(datasets['test'][pair[0]+'vsAbstract_BLEU'], datasets['test'][pair[1]+'vsAbstract_BLEU']))
    print('\n')