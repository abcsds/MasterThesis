import numpy as np
import csv
import spacy

model = {}
nlp = spacy.load("en_core_web_sm")

path = "./models/GloVe/glove.6B/glove.6B.300d.txt"

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        model[word] = vector

inpath = './data/CrowdFlower/text_emotion.csv'
# header = ['tweet_id', 'sentiment', 'author', 'content']

X = []
Y = []
with open(inpath, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sent = []
        for token in nlp(row['content']):  # TODO: tokenize
            try:
                sent += [model[token.text]]
            except KeyError:  # Word not in vocabulary
                # print(f"Error embedding: {word}")
                pass
        try:
            sent_avg = np.mean(sent, axis=0)
            assert sent_avg.shape[0] == 300, f"Sentence dimension is not 300: {word}"
            X += [sent_avg]
            Y += [row["sentiment"]]
        except IndexError:
            continue

assert len(X) == len(Y)
lens = [e.shape[0] for e in X]
assert len(set(lens)) == 1


# Because fasttext prefers to read from file
outpath = "./data/CrowdFlower/GloVe/embeddings.csv"

with open(outpath, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(X[0]))] + ['emotion']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, l in zip(X, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'emotion': l}))
