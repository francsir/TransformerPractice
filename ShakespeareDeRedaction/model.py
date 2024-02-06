from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

imdb_data = load_dataset("imdb")

#inspect the imdb dataset

print(imdb_data)