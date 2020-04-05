FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
ADD https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/book_corpus_wiki_en_uncased-a6607397.zip /root/.mxnet/models/
ADD https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/bert_12_768_12_book_corpus_wiki_en_uncased-75cc780f.zip /root/.mxnet/models/
RUN pip install mxnet-cu92 bert-embedding gensim seaborn pandas sklearn fasttext
