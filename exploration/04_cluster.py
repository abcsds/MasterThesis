import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})
dss = [
       "data/CrowdFlower/FastText/embeddings_unsupervised.csv",
       "data/CrowdFlower/FastText/embeddings_supervised.csv",
       "data/CrowdFlower/GloVe/embeddings.csv",
       "data/CrowdFlower/Word2Vec/embeddings.csv",
       # "data/EmotionPush/FastText/embeddings_unsupervised.csv",
       # "data/EmotionPush/FastText/embeddings_supervised.csv",
       "data/EmotionPush/GloVe/embeddings.csv",
       "data/EmotionPush/Word2Vec/embeddings.csv",
       "data/Friends/FastText/embeddings_unsupervised.csv",
       "data/Friends/FastText/embeddings_supervised.csv",
       "data/Friends/GloVe/embeddings.csv",
       "data/Friends/Word2Vec/embeddings.csv",
       ]
for i, ds in enumerate(dss):
    print(f"Working on {ds}")
    df = pd.read_csv(ds)
    X = df.drop("emotion", axis=1).to_numpy()
    Y = df["emotion"].to_numpy()
    assert X.shape[0] == Y.shape[0]
    ind = list(np.unique(Y))
    for emotion in ind:
        x = X[Y == emotion]
        try:
            p = sns.clustermap(np.corrcoef(x), figsize=(16, 16))
            p.fig.suptitle(f"Clustermap of {ds} {emotion}", fontsize=24)
            p.savefig(f"./img/cls/clustermap_{i}_{emotion}.png")
            del p
            plt.close()
        except FloatingPointError:
            continue

# i = 1
#
# i += 1
# print(i)
# ds = dss[i]
# print(f"Working on {ds}")
# df = pd.read_csv(ds)
# X = df.drop("emotion", axis=1).to_numpy()
# Y = df["emotion"].to_numpy()
# assert X.shape[0] == Y.shape[0]
# ind = list(np.unique(Y))
# for emotion in ind:
#     x = X[Y == emotion]
#     p = sns.clustermap(np.corrcoef(x), figsize=(16, 16))
#     p.fig.suptitle(f"Clustermap of {ds} {emotion}", fontsize=24)
#     p.savefig(f"./img/cls/clustermap_{i}_{emotion}.png")
#     del p
#     plt.close()
