import gensim
import csv
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
model_path = './models/Word2Vec/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

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
lens = [len(e) for e in X]
assert len(set(lens))==1

# Because fasttext prefers to read from file
outpath = "./data/CrowdFlower/Word2Vec/embeddings.csv"

with open(outpath, 'w', newline='') as f:
    fieldnames = [f"d{i}" for i in range(len(X[0]))] + ['emotion']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for e, l in zip(X, Y):
        writer.writerow(dict({f"d{i}": ei for i, ei in enumerate(e)}, **{'emotion': l}))
