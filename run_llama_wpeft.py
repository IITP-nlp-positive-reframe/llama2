# Import
import argparse
import pandas as pd
import numpy as np
import random
import os
import torch
import pdb
import csv
from datasets import load_dataset, load_metric
from sentence_transformers import SentenceTransformer, util
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from transformers import Trainer, pipeline, set_seed, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, prepare_model_for_kbit_training

from accelerate import Accelerator
accclerator = Accelerator()
device = "cuda" if torch.cuda.is_available() else "cpu"

import nltk
nltk.download('punkt')

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'false'
'''os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"'''

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling

## Data loader
train_path =  'data/wholetrain_gpt.txt'
dev_path = 'data/wholedev.csv'
test_path = 'data/wholetest.csv'
output_dir = 'output/'

metric = load_metric("rouge")
metric2 = load_metric("sacrebleu")

def run_unconstrained_llama2():
    
    ### Bit and Byte Config
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      load_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        device_map = "auto",
        trust_remote_code = True,
        quantization_config = bnb_config,
    )
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
    r=16,  # dimension of the updated matrices
    lora_alpha=64,  # parameter for scaling
    target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"],
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    ## Use llama from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.bos_token = "<startoftext>"
    tokenizer.eos_token = "<endoftext>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    from transformers import TextDataset

    def load_dataset(train_path,test_path,tokenizer):
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=train_path,
            block_size=50)
        test_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=test_path,
            block_size=50)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )
        return train_dataset,test_dataset,data_collator

    torch.cuda.empty_cache()
    train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)
    torch.cuda.empty_cache()

    '''def tokenization(example):
        return tokenizer(example["text"])
    train_dataset = dataset.map(tokenization,batched=True)'''

    batch_size = 8
    optim = "paged_adamw_32bit"
    per_device_train_batch_size = batch_size
    gradient_accumulation_steps = batch_size
    # save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 5
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        learning_rate=learning_rate,
        # per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        evaluation_strategy="no",
        do_train=True,
        do_eval=False,
        logging_steps=1024,
        save_steps=2048,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        # group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        #max_steps=1500, # delete for full training
        num_train_epochs = 10, #TRAIN_EPOCHS
        overwrite_output_dir=True,
        save_total_limit=3,
        fp16=True,
        eval_accumulation_steps=2
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        #pdb.set_trace()
        #print(predictions.shape)
        predictions = predictions.argmax(2)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #labels = labels.tolist()
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True,)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value.mid.fmeasure for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu scores
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        return {k: round(v, 4) for k, v in result.items()}
    
    trainer = transformers.Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        # train_dataset=tokenized_train_datasets["train"],
        # eval_dataset=tokenized_test_datasets["train"],
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    model.config.use_cache = False
    #model.to(device)
    
    trainer.train()
    torch.cuda.empty_cache()

    trainer.evaluate()

    torch.cuda.empty_cache()

    trainer.save_model("output/reframer")
    #pdb.set_trace()
    # Load trained model
    model = AutoModelForCausalLM.from_pretrained("output/reframer")
    model = prepare_model_for_kbit_training(model)
    config = PeftConfig.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('text-generation', model=model, tokenizer=tokenizer,eos_token_id=tokenizer.eos_token_id)

    import csv
    with open (test_path, newline="") as data:
      annotations = csv.DictReader(data, delimiter=',', quotechar='"')
      annotations_list = list(annotations)
      reframed_phrases = []
      answer_phrases = []
      for i in range(len(annotations_list)):
          prefix = "<|startoftext|> " + annotations_list[i]['original_text'] + "\nreframed:"
          gen_text = reframer(prefix, max_length=100)[0]['generated_text']
          reframed_phrases.append(gen_text)
          answer_phrases.append(annotations_list[i]['reframed_text'])

    # test = pd.read_csv(test_path)
    # texts = test['original_text'].to_list()
    # reframed_phrases = [reframer(phrase)[0]['generated_text'] for phrase in texts]
    root = "./"
    with open('llama2_7b_predict.txt', 'w') as f:
        for item in reframed_phrases:
            print(item)
            f.write("%s\n" % item)

    with open("total_reframec.txt",'w') as f:
      for item in answer_phrases:
        print(item)
        f.write("%s\n"%item)
import gc
torch.cuda.empty_cache()
gc.collect()
    
run_unconstrained_llama2()
