import fasttext
import csv
import numpy as np
import spacy

inpath = "./data/CrowdFlower/text_emotion.csv"
header = ["tweet_id", "sentiment", "author", "content"]
nlp = spacy.load("en_core_web_sm")

X = []
Y = []
with open(inpath, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        X += [row["content"]]
        Y += [row["sentiment"]]

# Because fasttext prefers to read from file
outpath = "./data/CrowdFlower/FastText/text.txt"
with open(outpath, 'w') as f:
    # f.write("\n".join(X))
    for item in X:
        f.write(f"{item}\n")


# Unsupervised
# Skipgram model
model = fasttext.train_unsupervised(outpath, model='skipgram')
embeddings = [np.mean([model[token.text] for token in nlp(tweet)], axis=0) for tweet in X]
assert len(embeddings) == len(Y)
lens = [len(e) for e in embeddings]
assert len(set(lens))==1

embeddings_path = "./data/CrowdFlower/FastText/embeddings_unsupervised.csv"

with open(embeddings_path, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(embeddings[9]))] + ['sentiment']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, s in zip(embeddings, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'sentiment': s}))


# Supervised

outpath = "./data/CrowdFlower/FastText/text_sup.txt"
with open(outpath, 'w') as f:
    for item, label in zip(X, Y):
        f.write(f"{item} __label__{label}\n")
model = fasttext.train_supervised(outpath)

embeddings = [np.mean([model[token.text] for token in nlp(tweet)], axis=0) for tweet in X]

assert len(embeddings) == len(Y)
lens = [len(e) for e in embeddings]
assert len(set(lens))==1

embeddings_path = "./data/CrowdFlower/FastText/embeddings_supervised.csv"

with open(embeddings_path, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(embeddings[9]))] + ['sentiment']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, s in zip(embeddings, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'sentiment': s}))
