import csv
import numpy as np
import pandas as pd
import spacy
import gensim

# GloVe
model = {}
nlp = spacy.load("en_core_web_sm")

path = "./models/GloVe/glove.6B/glove.6B.300d.txt"

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        model[word] = vector

inpath = './models/EmoLex/dict.csv'
# header = ["Word","anger","anticipation","disgust",
#           "fear","joy","negative","positive",
#           "sadness","surprise","trust"]
emotions = ['anger', 'anticipation', 'disgust',
            'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust']
X = []
Y = []
with open(inpath, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels = [int(row[e]) for e in emotions]
        if sum(labels) > 1:
            try:
                X.append(model[row["Word"]])
                Y.append(labels)
            except KeyError:  # Word not in vocabulary
                # print(f"Error embedding: {word}")
                pass
X = np.array(X)
Y = np.array(Y)
assert X.shape[0] == Y.shape[0]



# Word2Vec
nlp = spacy.load("en_core_web_sm")
model_path = './models/Word2Vec/GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

inpath = './data/CrowdFlower/text_emotion.csv'
# header = ['tweet_id', 'sentiment', 'author', 'content']
