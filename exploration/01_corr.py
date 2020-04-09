import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize


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

    cors = []
    for emotion in ind:
        y = (Y == emotion).astype(int)
        cor_p_sent = []
        for j in range(X.shape[1]):
            x = normalize(X[:, j].reshape(-1, 1)).reshape(-1)
            c = np.corrcoef(x, y)[1,0]
            if c > .80:
                print(f"Emotion: {emotion}, Dimension: {j}, corrcoef: {c}")
            cor_p_sent.append(c)
        cors.append(cor_p_sent)
    cors = np.array(cors)
    assert np.isfinite(cors).all() == True, "Some infinites or nan in correlations"

    x = cors[(np.isfinite(cors))]
    x = np.nan_to_num(cors)
    assert np.isfinite(x).all() == True, "Some infinites or nan in correlations"

    g = sns.clustermap(x, col_cluster=False)
    t = [int(tick_label.get_text()) for tick_label in g.ax_heatmap.axes.get_yticklabels()]
    sorted_e = [x for _,x in sorted(zip(t,ind))]
    g.ax_heatmap.axes.set_yticklabels(sorted_e,  rotation=0)
    g.fig.suptitle(f"Correlation of {ds}", fontsize=18)
    g.savefig(f"./img/cor/correlation_{i}.png")
