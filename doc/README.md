# Master Thesis
To be compiled with pdflatex:

```
pdflatex -output-directory aux thesis.tex
```

Using [atom latex](https://atom.io/packages/latex) package, and following suggestions from the package [author](https://gist.github.com/Aerijo/5b9522530715e5be6e89fc012e9a72a8)


# Revisiion notes
 1. The first half of the thesis is quite hard to read, since everything is quite abstract and it is no clear what you are actually going to do. Maybe at some point in the beginning you can just say that you want to find out, whether some dimension of a word embedding represents an emotion. That is a bit simplistic, but such a simple idea can help as a guide when reading all the details on the methods, data sets etc.

 2. When you describe the datasets you might give a little bit more information on them E.g. the number of labels for each class, average sentence length, etc. Most important: give a few examples! Then the readers get an idea what the data look like and what we expect the algorithms to do.

 3. You give many details on hardware and software libraries you have used. Is that a requirement from the Department? Usually one would give more information on the methods using a abstract (mathematical) representation, using formula, and leave the details about the implementation hidden.

 4. In the end you could give a table with 8 columns (for each dimension) and 4 rows (for W2V, FastText, Glove and Bert) and in each row put two numbers: (1) the number of dimension that positively or negatively correlate with that dimension (i.e. that have a value above the threshold) and (2) the correlation with the highest correlating dimension, again only if it is above the threshold. That will result in a rather sparse matrix, but it would give all the results in one table. Since PCA didn’t help, it is enough to make the table for the original dimensions. However, now you could also give the table for the other emotion datasets (EmotionPush and Friends). It would be too much to give all the visualizations for these datasets, but one small table for each data set would add a lot of value.
