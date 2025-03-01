import json
import numpy as np
import fasttext
import csv
import spacy


nlp = spacy.load("en_core_web_sm")
path1 = "./data/EmotionPush/emotionpush.dev.json"
path2 = "./data/EmotionPush/emotionpush.test.json"
path3 = "./data/EmotionPush/emotionpush.train.json"

Y = []
X = []
data = {}
for path in [path1, path2, path3]:
    with open(path, 'r') as f:
        json_data = f.read()
        data = json.loads(json_data)

    for doc in data:
        for sent in doc:
            Y.append(sent["emotion"])
            X.append(sent["utterance"])
assert len(X) == len(Y)

# Unsupervised

# Because fasttext prefers to read from file
outpath = "./data/EmotionPush/FastText/text.txt"
with open(outpath, 'w') as f:
    # f.write("\n".join(X))
    for item in X:
        f.write(f"{item}\n")

# Skipgram model
model = fasttext.train_unsupervised(outpath, model='skipgram')
# TODO: some embeddings don't have dimension
embeddings = [np.mean([model[token.text] for token in nlp(doc)], axis=0) for doc in X]
assert len(embeddings) == len(Y)
lens = [e.shape[0] for e in embeddings]
assert len(set(lens))==1

embeddings_path = "./data/EmotionPush/FastText/embeddings_unsupervised.csv"

with open(embeddings_path, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(embeddings[9]))] + ['emotion']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, l in zip(embeddings, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'emotion': l}))


# Supervised

outpath = "./data/EmotionPush/FastText/text_sup.txt"
with open(outpath, 'w') as f:
    for item, label in zip(X, Y):
        f.write(f"{item} __label__{label}\n")
model = fasttext.train_supervised(outpath)

embeddings = [np.mean([model[token.text] for token in nlp(doc)], axis=0) for doc in X]

assert len(embeddings) == len(Y)
lens = [len(e) for e in embeddings]
assert len(set(lens))==1

embeddings_path = "./data/EmotionPush/FastText/embeddings_supervised.csv"

with open(embeddings_path, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(embeddings[9]))] + ['emotion']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, l in zip(embeddings, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'emotion': l}))
