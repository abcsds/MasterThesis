import numpy as np
import json
import csv
import spacy


nlp = spacy.load("en_core_web_sm")
model = {}
path = "./models/GloVe/glove.6B/glove.6B.300d.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        model[word] = vector


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
            embd = []
            for token in nlp(sent["utterance"]):
                try:
                    embd += [model[token.text]]
                except KeyError:  # Word not in vocabulary
                    pass
            try:
                if len(embd) == 0:
                    continue
                sent_avg = np.mean(embd, axis=0)
                assert sent_avg.shape[0] == 300, "Sentence dimension is not 300"
                X += [sent_avg]
                Y.append(sent["emotion"])
            except (TypeError, IndexError):
                continue

assert len(X) == len(Y)

# Because fasttext prefers to read from file
outpath = "./data/EmotionPush/GloVe/embeddings.csv"

with open(outpath, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(X[0]))] + ['emotion']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, l in zip(X, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'emotion': l}))
