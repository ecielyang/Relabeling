from Smallest_k import IP
# We use spacy to get feature vectors for our input text.
import numpy as np
from tqdm import tqdm
import spacy

sst_dataset = {}
for split in ["train", "dev", "test"]:
    URL = f"https://raw.githubusercontent.com/successar/instance_attributions_NLP/master/Datasets/SST/data/{split}.jsonl"
    import urllib.request, json
    with urllib.request.urlopen(URL) as url:
        data = url.read().decode()
        data = [json.loads(line) for line in data.strip().split("\n")]
        sst_dataset[split] = data


nlp = spacy.load('en_core_web_md')

X, y = {}, {}
for split in ["train", "dev"]:
    X[split] = np.array([nlp(example["document"]).vector for example in tqdm(sst_dataset[split])])
    y[split] = np.array([example["label"] for example in sst_dataset[split]])

thresh = 0.5
l2 = 1000

IP(X, y, l2, "SST", thresh)