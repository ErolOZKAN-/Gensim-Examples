import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.corpora import Dictionary
from gensim import corpora, models, similarities

dictionary = Dictionary.load("../output/dictionary.dict")  # Load a dictionary

## MEMORY EFFICIENT WAY TO CREATE CORPUS
class MyCorpus(object):
    def __iter__(self):
        for line in open('../data/mycorpus'):
            yield dictionary.doc2bow(line.lower().split())

corpus = MyCorpus()  # doesn't load the corpus into memory!

# step 1 -- initialize a model
tfidf = models.TfidfModel(corpus)

# convert a document to tfidf reprentation
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])

#convert all corpus to tdids reprenstation
corpusTfidf = tfidf[corpus]
for doc in corpusTfidf:
    print (doc)

# initialize an LSI transformation
lsi = models.LsiModel(corpusTfidf, id2word=dictionary, num_topics=2)

# create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
corpus_lsi = lsi[corpusTfidf]
lsi.print_topics(2)

# both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
for doc in corpus_lsi:
    print(doc)

# save and load model into/from disk
lsi.save('../output/model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('../output/model.lsi')