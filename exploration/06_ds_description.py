import pandas as pd


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

for ds in dss:
    df = pd.read_csv(ds)
    print(ds, len(df.columns) - 1, len(df["emotion"].unique()))
    print(df["emotion"].unique())
    
