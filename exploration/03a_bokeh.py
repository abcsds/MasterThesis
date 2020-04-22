import numpy as np
import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE
# from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
from bokeh.plotting import figure, output_file, show

dss = ["data/CrowdFlower/FastText/embeddings_unsupervised.csv",
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
    n_classes = len(np.unique(Y))
    palette = viridis(n_classes)
    ind = list(np.unique(Y))
    projection = TSNE(n_jobs=6).fit_transform(X)

    # output to static HTML file
    output_file(f"img/int/scatter_{i}.html")

    source = ColumnDataSource(dict(
        x=projection[:, 0],
        y=projection[:, 1],
        color=[palette[ind.index(i)] for i in Y],
        label=Y
    ))

    tooltips = [("index", "$index"),
                ("(x,y)", "($x, $y)"),
                ("label", "@label"),
                ]
    # create a new plot
    p = figure(tools="pan,box_zoom,reset,save,hover",
               y_range=[-60, 60], x_range=[-40, 40],
               tooltips=tooltips,
               plot_width=1600, plot_height=800,
               title=f"TSNE for {ds}",
               x_axis_label='X', y_axis_label='Y'
               )
    # legend field matches the column in the source
    p.circle(x='x', y='y',
             radius=0.1,
             color='color',
             legend_group='label',
             source=source)
    # show the results
    show(p)
    del p
