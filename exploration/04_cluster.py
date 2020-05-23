import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


sns.set_style('darkgrid')
sns.set_palette('muted')
enc = OneHotEncoder(sparse=False)

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
    ind = list(np.unique(Y))
    # for emotion in ind:
    #     x = X[Y == emotion]
    #     try:
    #         p = sns.clustermap(np.corrcoef(x), figsize=(16, 16))
    #         p.fig.suptitle(f"Clustermap of {ds} {emotion}", fontsize=24)
    #         p.savefig(f"./img/cls/cm_{data}_{model}_{emotion}.png")
    #         del p
    #         plt.close()
    #     except FloatingPointError:
    #         continue

    # Emotion to emotion correlation
    enc.fit(np.unique(Y).reshape(-1,1))
    mat = enc.transform(Y.reshape(-1,1))
    cor = np.corrcoef(mat.transpose())
    p = sns.clustermap(cor, figsize=(16, 16))
    t = [int(tick_label.get_text()) for tick_label in p.ax_heatmap.axes.get_yticklabels()]
    sorted_e = [x for _,x in sorted(zip(t,ind))]
    p.ax_heatmap.axes.set_yticklabels(sorted_e,  rotation=0)
    p.fig.suptitle(f"Clustermap of {ds} E-E", fontsize=24)
    p.savefig(f"./img/cls/cmE_{data}_{model}.png")
    del p
    plt.close()
