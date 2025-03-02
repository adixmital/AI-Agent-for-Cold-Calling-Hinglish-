# Dataset loading and fine-tuning
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,TrainingArguments, Trainer, pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import os
from datasets import load_dataset
dataset = load_dataset("json", data_files="hinglish_cold_calls.jsonl")


def tokenize_function(example):
    tokens = tokenizer(
        example["instruction"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens



dataset = dataset["train"].train_test_split(test_size=0.2)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./cold_call_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

# Generate Hinglish text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

