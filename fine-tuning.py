from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

##define the trainging arguments, test_trainer is the name of the folder where the model will be saved
training_args = TrainingArguments("test_trainer")


raw_dataset = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)