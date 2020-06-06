import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


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
    projection = PCA().fit_transform(X)

    cors = []
    for emotion in ind:
        y = (Y == emotion).astype(int)
        cor_p_sent = []
        for j in range(projection.shape[1]):
            x = normalize(projection[:, j].reshape(-1, 1)).reshape(-1)
            c = np.corrcoef(x, y)[1,0]
            if c > .40:
                print(f"Emotion: {emotion}, Dimension: {j}, corrcoef: {c}")
            cor_p_sent.append(c)
        cors.append(cor_p_sent)
    cors = np.array(cors)
    x = np.nan_to_num(cors)
    assert np.isfinite(x).all() == True, "Some infinites or nan in correlations"
    if np.isfinite(x).all() != True:
        continue

    print(f"Correlation Maximum: {np.amax(x)}")
    max_emo, max_dim = np.where(x == np.amax(x))
    max_emo = ind[max_emo[0]]
    print(f"Maximum emotion and dim: {max_emo}, {max_dim[0]}")

    # correlation
    g = sns.clustermap(x, col_cluster=False)
    t = [int(tick_label.get_text()) for tick_label in g.ax_heatmap.axes.get_yticklabels()]
    sorted_e = [x for _,x in sorted(zip(t,ind))]
    g.ax_heatmap.axes.set_yticklabels(sorted_e,  rotation=0)
    g.fig.suptitle(f"Correlation of PCA {data}/{model}", fontsize=18)
    g.savefig(f"./img/pca/pca_cor_{data}_{model}.png")
    plt.close()

    # scatter
    da = pd.DataFrame(list(zip(projection[:, 0],projection[:, 1],Y)),
                        columns=["Component0","Component1", "Emotion"])
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sns.scatterplot(ax=ax,
                    x="Component0", y="Component1",
                    hue="Emotion",
                    data=da)
    fig.suptitle(f"PCA of {ds}", fontsize=18)
    fig.savefig(f"./img/pca/scat_{data}_{model}.png")
    del(fig)
