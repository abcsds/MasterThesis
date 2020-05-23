import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import seaborn as sns


sns.set_style('darkgrid')
sns.set_palette('muted')

dss = [
       "data/CrowdFlower/FastText/embeddings_unsupervised.csv",
       "data/CrowdFlower/FastText/embeddings_supervised.csv",
       "data/CrowdFlower/GloVe/embeddings.csv",
       "data/CrowdFlower/Word2Vec/embeddings.csv",
       "data/CrowdFlower/BERT/embeddings.csv",
       # "data/EmotionPush/FastText/embeddings_unsupervised.csv",
       # "data/EmotionPush/FastText/embeddings_supervised.csv",
       "data/EmotionPush/GloVe/embeddings.csv",
       "data/EmotionPush/Word2Vec/embeddings.csv",
       "data/EmotionPush/BERT/embeddings.csv",
       "data/Friends/FastText/embeddings_unsupervised.csv",
       "data/Friends/FastText/embeddings_supervised.csv",
       "data/Friends/GloVe/embeddings.csv",
       "data/Friends/Word2Vec/embeddings.csv",
       "data/Friends/BERT/embeddings.csv",
       ]

for i, ds in enumerate(dss):
    data, model = os.path.split(ds)[0].split("/")[1:3]
    print(f"Working on {data}/{model}")
    df = pd.read_csv(ds)
    df = df[~df["emotion"].isin(["neutral", "non-neutral", "empty"])]
    X = df.drop("emotion", axis=1).to_numpy()
    Y = df["emotion"].to_numpy()
    assert X.shape[0] == Y.shape[0]
    n_classes = len(np.unique(Y))
    palette = np.array(sns.color_palette("hls", n_classes))
    ind = list(np.unique(Y))
    projection = TSNE(n_jobs=6).fit_transform(X)
    fig = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(projection[:, 0], projection[:, 1],
                    c=palette[[ind.index(i) for i in Y]],
                    label=ind
                   )
    fig.suptitle(f"TSNE of {data}/{model}", fontsize=24)
    fig.savefig(f"./img/tsne/scat_{data}_{model}.png")
