import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from MulticoreTSNE import MulticoreTSNE as TSNE
# from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, show


# ======================= Load Model: GloVe
model = {}
path = "./models/GloVe/glove.6B/glove.6B.300d.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        model[word] = vector


# ====================== Load Data: Emolex
inpath = "./models/EmoLex/dict.csv"
# header = ["Word","anger","anticipation","disgust",
#           "fear","joy","negative","positive",
#           "sadness","surprise","trust"]
emotions = ['anger', 'anticipation',
            'disgust', 'fear', 'joy',
            'sadness', 'surprise', 'trust']

d = pd.read_csv(inpath)
# cleanup
# d.loc[d['anger'] != 0]
df = d.loc[(d['anger'] != 0) |
           (d['anticipation'] != 0) |
           (d['disgust'] != 0) |
           (d['fear'] != 0) |
           (d['joy'] != 0) |
           (d['sadness'] != 0) |
           (d['surprise'] != 0) |
           (d['trust'] != 0) ].drop(["positive", "negative"], axis=1)
df = df.loc[df.sum(axis=1) == 1]


# ====================== Embed data
W, X, Y = [],[],[]
for word in df["Word"].values:
    W.append(word)
    X.append(model[word])
    enc_emo = df[df["Word"]==word].drop("Word", axis=1).values[0]
    Y.append(emotions[np.where(enc_emo == 1)[0][0]])
W = np.array(W)
X = np.array(X)
Y = np.array(Y)
assert X.shape[0] == Y.shape[0]


# ========================== Linear Correlation
cors = []
for emotion in emotions:
    y = (Y == emotion).astype(int)
    cor_p_sent = []
    for j in range(X.shape[1]):
        x = normalize(X[:, j].reshape(-1, 1)).reshape(-1)
        c = np.corrcoef(x, y)[1,0]
        if c > .40:
            print(f"Emotion: {emotion}, Dimension: {j}, corrcoef: {c}")
        cor_p_sent.append(c)
    cors.append(cor_p_sent)
cors = np.array(cors)
x = np.nan_to_num(cors)
assert np.isfinite(x).all() == True, "Some infinites or nan in correlations"

sns.color_palette("deep")
g = sns.clustermap(x, col_cluster=False, vmin=-.5, vmax=.5, cmap=plt.get_cmap("seismic"))
t = [int(tick_label.get_text()) for tick_label in g.ax_heatmap.axes.get_yticklabels()]
ind = list(np.unique(Y))
sorted_e = [x for _,x in sorted(zip(t,ind))]
g.ax_heatmap.axes.set_yticklabels(sorted_e,  rotation=0)
g.fig.suptitle(f"Correlation of Emolex/GloVe", fontsize=18)
g.savefig(f"./img/emolex/cor_emolex_GloVe.png")
plt.close()
print("====== linear")
print(f"Correlation Maximum: {np.amax(x)}")
max_emo, max_dim = np.where(x == np.amax(x))
max_emo = emotions[max_emo[0]]
print(f"Maximum emotion and dim: {max_emo}, {max_dim[0]}")

#========================== PCA
projection = PCA().fit_transform(X)
# n_classes = len(np.unique(Y))
# palette = np.array(sns.color_palette("hls", n_classes))
# palette = Category10[n_classes]

ind = list(np.unique(Y))
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

print("====== pca")
print(f"Correlation Maximum: {np.amax(x)}")
max_emo, max_dim = np.where(x == np.amax(x))
max_emo = emotions[max_emo[0]]
print(f"Maximum emotion and dim: {max_emo}, {max_dim[0]}")

# PCA correlation clustermap
g = sns.clustermap(x, col_cluster=False, vmin=-.5, vmax=.5, cmap=plt.get_cmap("seismic"))
t = [int(tick_label.get_text()) for tick_label in g.ax_heatmap.axes.get_yticklabels()]
sorted_e = [x for _, x in sorted(zip(t, ind))]
g.ax_heatmap.axes.set_yticklabels(sorted_e,  rotation=0)
g.fig.suptitle(f"Correlation of PCA Emolex/GloVe", fontsize=18)
g.savefig(f"./img/emolex/pca_cor_emolex_glove.png")
plt.close()

# PCA minimized correlation clustermap
x_r = x[:, 0:len(ind)]
g = sns.clustermap(x_r, annot=x_r, col_cluster=False, vmin=-.5, vmax=.5, cmap=plt.get_cmap("BrBG"))
t = [int(tick_label.get_text()) for tick_label in g.ax_heatmap.axes.get_yticklabels()]
sorted_e = [x for _, x in sorted(zip(t, ind))]
g.ax_heatmap.axes.set_yticklabels(sorted_e,  rotation=0)
g.fig.suptitle(f"Correlation of PCA Emolex/GloVe (reduced)", fontsize=18)
g.savefig(f"./img/emolex/pca_cor_emolex_glove_min.png")
plt.close()


# Static scatter plot
data = pd.DataFrame(list(zip(projection[:, 0],projection[:, 1],Y)),
                    columns=["Component0","Component1", "Emotion"])
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sns.scatterplot(ax=ax,
                x="Component0", y="Component1",
                hue="Emotion",
                data=data)
fig.suptitle(f"PCA of Emolex/GloVe", fontsize=24)
fig.savefig(f"./img/emolex/pca_scat_emolex_glove.png")

# Interactive scatter plot
n_classes = len(np.unique(Y))
palette = Category10[n_classes]

# output to static HTML file
output_file(f"img/emolex/pca_scat_emolex.html")

source = ColumnDataSource(dict(
    x=projection[:, 0],
    y=projection[:, 1],
    color=[palette[ind.index(i)] for i in Y],
    label=Y,
    text=W
))

tooltips = [("index", "$index"),
            ("text", "@text"),
            ("(x,y)", "($x, $y)"),
            ("label", "@label"),
            ]
# create a new plot
p = figure(tools="pan,box_zoom,reset,save,hover",
           # y_range=[-4, 4], x_range=[-3, 4],
           tooltips=tooltips,
           plot_width=1600, plot_height=800,
           title=f"TSNE for Emolex/GloVe",
           x_axis_label='X', y_axis_label='Y'
           )
# legend field matches the column in the source
p.circle(x='x', y='y',
         radius=0.05,
         color='color',
         legend_group='label',
         source=source)
# show the results
show(p)

#========================== TSNE
projection = TSNE(n_jobs=6).fit_transform(X)

# correlation
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

print("====== tsne")
print(f"Correlation Maximum: {np.amax(x)}")
max_emo, max_dim = np.where(x == np.amax(x))
max_emo = emotions[max_emo[0]]
print(f"Maximum emotion and dim: {max_emo}, {max_dim[0]}")


# clustermap tsne correlation
g = sns.clustermap(x, col_cluster=False, annot=x)
t = [int(tick_label.get_text()) for tick_label in g.ax_heatmap.axes.get_yticklabels()]
sorted_e = [x for _,x in sorted(zip(t,ind))]
g.ax_heatmap.axes.set_yticklabels(sorted_e,  rotation=0)
g.fig.suptitle(f"Correlation of TSNE Emolex/GloVe", fontsize=18)
g.savefig(f"./img/emolex/tsne_cor_emolex_glove.png")
plt.close()

# Static scatter plot
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
# Static scatter plot
data = pd.DataFrame(list(zip(projection[:, 0],projection[:, 1],Y)),
                    columns=["Component0","Component1", "Emotion"])
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')
sns.scatterplot(ax=ax,
                x="Component0", y="Component1",
                hue="Emotion",
                data=data)
fig.suptitle(f"TSNE of Emolex/GloVe", fontsize=24)
fig.savefig(f"./img/emolex/tsne_scat_emolex_glove.png")


n_classes = len(np.unique(Y))
palette = Category10[n_classes]
ind = list(np.unique(Y))

# Interactive TSNE
# output to static HTML file
output_file(f"img/emolex/tsne_scat_emolex.html")

source = ColumnDataSource(dict(
    x=projection[:, 0],
    y=projection[:, 1],
    color=[palette[ind.index(i)] for i in Y],
    label=Y,
    text=W
))

tooltips = [("index", "$index"),
            ("text", "@text"),
            ("(x,y)", "($x, $y)"),
            ("label", "@label"),
            ]
# create a new plot
p = figure(tools="pan,box_zoom,reset,save,hover",
           y_range=[-30, 30], x_range=[-30, 30],
           tooltips=tooltips,
           plot_width=1600, plot_height=800,
           title=f"TSNE for Emolex/GloVe",
           x_axis_label='X', y_axis_label='Y'
           )
# legend field matches the column in the source
p.circle(x='x', y='y',
         radius=0.3,
         color='color',
         legend_group='label',
         source=source)
# show the results
show(p)


# ============================== Clustermap of emotion correlation
x = df.drop("Word", axis=1).corr()
g = sns.clustermap(x, annot=x, vmin=-.5, vmax=.5, cmap=plt.get_cmap("seismic"))
g.ax_col_dendrogram.set_visible(False)
g.fig.suptitle(f"Correlation of emotion-emotion Emolex/GloVe", fontsize=20)
g.savefig(f"./img/emolex/cor_emolex_GloVe_e_e.png")
plt.close()
