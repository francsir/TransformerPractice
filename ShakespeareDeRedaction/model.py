from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
import collections
import numpy as np
import math

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

sonnet_dataset = load_from_disk('./ShakespeareDeRedaction/sonnets_dataset')


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_datasets = sonnet_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

chunk_size = 128
print(tokenized_datasets.keys())

def group_texts(examples):

    concatenated_examples = {k:sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length//chunk_size) * chunk_size

    result = {
        k:[t[i:i+chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_datasets.map(group_texts, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)
        mask = np.random.binomial(1, wwm_probability, (len(mapping), ))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            for idx in mapping[word_id]:
                new_labels[idx] = input_ids[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels
    return data_collator(features)



batch_size = 64

logging_steps = len(lm_dataset["train"])
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
	output_dir = f"{model_name}-finetuned=shakespeareSonnets",
	overwrite_output_dir = True,
	evaluation_strategy="epoch",
	learning_rate = 2e-5,
	weight_decay = 0.01,
	per_device_train_batch_size = batch_size,
	per_device_eval_batch_size = batch_size,
	push_to_hub = False,
	fp16 = True,
	logging_steps= logging_steps,
	remove_unused_columns = False,
)

trainer = Trainer(
	model=model,
	args = training_args,
	train_dataset = lm_dataset["train"],
	eval_dataset = lm_dataset["test"],
	data_collator = whole_word_masking_data_collator,
	tokenizer=tokenizer,
)

import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



        	