import torch 
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

##Training a basic model using HuggingFace bert-base-uncased, with 2 

## Checkpoints are pretrained weights that can be used to initialize a model when training a new model.
checkpoint = "bert-base-uncased"
## The tokenizer is used to convert text into tokens that can be fed into the model. i.e it converts text into numbers. 
## AutoTokenizer finds the correct tokenizer based on the checkpoint name, and loads the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
## AutoModel finds the correct model based on the checkpoint name, and loads the weights into the model.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

##Sequences is a list of strings that we want to classify, with this model positive/negative sentiment.
sequences = ["I've been waiting for a HuggingFace course my whole life.", "This course is amazing!"]
## The tokenizer converts the sequences into tokens that can be fed into the model.

batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

## batch["labels"] is a tensor of labels that correspond to the sequences.
batch["labels"] = torch.tensor([1, 1])

## The model outputs a tuple of (loss, logits). The loss is used to update the weights of the model.
optimizer = AdamW(model.parameters())
#**batch unpacks the dictionary into keyword arguments for the model.
loss = model(**batch).loss
## The loss.backward() function calculates the gradients of the loss with respect to the model weights.
loss.backward()
## The optimizer.step() function updates the model weights with the calculated gradients.
optimizer.step()