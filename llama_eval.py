
from datasets import load_dataset, load_metric
from sentence_transformers import SentenceTransformer, util
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from transformers import Trainer, pipeline, set_seed, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, prepare_model_for_kbit_training
test_path = 'data/wholetest.csv'
model = AutoModelForCausalLM.from_pretrained("output/reframer")
model = prepare_model_for_kbit_training(model)
config = PeftConfig.from_pretrained("output/reframer")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained("output/reframer")
reframer = pipeline('text-generation', model=model, tokenizer=tokenizer,eos_token_id=tokenizer.eos_token_id)

import csv
from tqdm import tqdm
import pdb
with open (test_path, newline="") as data:
    annotations = csv.DictReader(data, delimiter=',', quotechar='"')
    annotations_list = list(annotations)
    reframed_phrases = []
    answer_phrases = []
    for i in tqdm(range(len(annotations_list))):
        #prefix = "<|startoftext|> " + annotations_list[i]['original_text'] + "\nreframed:"
        prefix = annotations_list[i]['original_text']
        gen_text = reframer(prefix, max_length=100)[0]['generated_text']
        init = gen_text.find('reframed:')
        end = gen_text.find('<end')
        if init==-1 or end== -1 or end-init<9:
            continue
            reframed_phrases.append(" ")
        else:
            reframed_phrases.append(gen_text[init+10:end])
        answer_phrases.append(annotations_list[i]['reframed_text'])

root = "./"
with open('predict.txt', 'w') as f:
    for item in reframed_phrases:
        #print(item)
        f.write("%s\n" % item)

with open("answer.txt",'w') as f:
    for item in answer_phrases:
        #print(item)
        f.write("%s\n"%item)
import argparse
from nlgeval import compute_individual_metrics
import pandas as pd
# from rouge_score import rouge_scorer
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
#from bert_score import score
import os
from nlgeval import NLGEval
import pdb
# from cal_eval import *
# from perspective import PerspectiveAPI
import time
# from cal_novelty import *
import torch
from tqdm import tqdm
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, GPT2Tokenizer
from datasets import load_dataset, load_metric

import evaluate

rouge = Rouge()

### SACREBLEU ###
sacrebleu = evaluate.load("sacrebleu")

predicts = reframed_phrases
reference = answer_phrases

results = sacrebleu.compute(predictions = predicts, references = reference)

# sacrebleu = metric2.compute(predictions=df['output'], references=[df['gt']])
# print(sacrebleu['score'])

### ROUGE ###
scores = rouge.get_scores(predicts, reference, avg=True)



# print(f"path : {args.file}")
# print("Toxicity score: ", df['TOXICITY'].mean())
# print('bleu-2:', metrics_dict['Bleu_2'] * 100)
print('sacrebleu:', results['score'])
print('rouge-1:', scores['rouge-1']['f'] * 100)
print('rouge-2:', scores['rouge-2']['f'] * 100)
print('rouge-l:', scores['rouge-l']['f'] * 100)


data_dict = {
    # 'bleu-2': metrics_dict['Bleu_2'] * 100,
    'sacrebleu': results['score'],
    'rouge-1': scores['rouge-1']['f'] * 100,
    'rouge-2': scores['rouge-2']['f'] * 100,
    'rouge-l': scores['rouge-l']['f'] * 100,
    
}

js = json.dumps(data_dict, indent=4)
with open("eval.json", "w") as outfile:
    outfile.write(js)



# test = pd.read_csv(test_path)
# texts = test['original_text'].to_list()
# reframed_phrases = [reframer(phrase)[0]['generated_text'] for phrase in texts]