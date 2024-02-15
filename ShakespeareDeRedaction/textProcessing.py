import re
from datasets import Dataset
from sklearn.model_selection import train_test_split

def clean(text):
    pattern = r"\b(?=[MDCLXVIΙ])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})([IΙ]X|[IΙ]V|V?[IΙ]{0,3})\b\.?"
    return re.sub(pattern, '', text)

def lowercase(text):
    return text.lower()

with open('./ShakespeareDeRedaction/complete_works.txt', 'r') as file:
    data = file.read()

numerals_removed = clean(data)

sonnets = [p.strip() for p in numerals_removed.split('\n\n') if p.strip()]

sentences = [sentence.strip() for sonnet in sonnets for sentence in sonnet.split('\n') if sentence]

sonnets_dataset = Dataset.from_dict({"text": sentences})

texts = sonnets_dataset["text"]

sonnets_dataset = Dataset.from_dict({"text": texts})

sonnets_dataset = sonnets_dataset.train_test_split(test_size=0.1, seed = 42)

##test_texts = sonnets_dataset["test"]["text"]
##val_texts, test_texts = train_test_split(test_texts, test_size=0.7, random_state=42)

##val_dataset= Dataset.from_dict({"text": val_texts})
##test_dataset= Dataset.from_dict({"text": test_texts})
##
##sonnets_dataset['test'] = test_dataset
##sonnets_dataset['validation'] = val_dataset

print(sonnets_dataset)
sonnets_dataset.save_to_disk('./ShakespeareDeRedaction/complete_works_dataset')






