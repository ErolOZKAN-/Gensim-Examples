import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.corpora import Dictionary
from gensim import models,similarities,corpora

#LOAD DICTIONARY AND CORPUS FROM DISK
dictionary = corpora.Dictionary.load('../output/dictionary.dict')
corpus = corpora.MmCorpus('../output/corpus.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

#CONVERT TO LSI VECTOR SPACE REP.
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

# GET LSI REP OF DOC
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsi[corpus])

# perform a similarity query against the corpus
sims = index[vec_lsi]

# print (document_number, document_similarity) 2-tuples
print(list(enumerate(sims)))

# print sorted (document number, similarity score) 2-tuples
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)

# SAVE AND LOAD FROM/INTO DISK
index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')