import gensim
import csv
import numpy as np
import json
import spacy


nlp = spacy.load("en_core_web_sm")
model_path = './models/Word2Vec/GoogleNews-vectors-negative300.bin.gz'

model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

path = "./data/Friends/Friends/friends.json"

with open(path, 'r') as f:
    json_data = f.read()
    data = json.loads(json_data)

Y = []
X = []
words = []
for doc in data:
    for sent in doc:
        # X.append(sent["utterance"])
        embd = []
        for token in nlp(sent["utterance"]):  # TODO: tokenize
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
outpath = "./data/Friends/Word2Vec/embeddings.csv"

with open(outpath, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(X[0]))] + ['emotion']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, l in zip(X, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'emotion': l}))
