from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

##loading the mrpc dataset from the glue tasks, will be split into train val and test
raw_dataset = load_dataset("glue", "mrpc")

# Take the train data set for inspection
raw_train_dataset = raw_dataset["train"]
print(raw_train_dataset[0])

##find the features of the dataset, and what they mean
print(raw_train_dataset.features)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


## function that will tokenize the data and keep is as a dataset, 
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

sample = tokenized_datasets["train"][:8]
sample = {k: v for k, v in sample.items() if k not in ['idx', 'sentence1', 'sentence2']}

batch = data_collator(sample)
print({k:v.shape for k, v in batch.items()})