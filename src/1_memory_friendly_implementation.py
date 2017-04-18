import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.corpora import Dictionary

dictionary = Dictionary.load("../output/dictionary.dict")  # Load a dictionary

## MEMORY EFFICIENT WAY TO CREATE CORPUS
class MyCorpus(object):
    def __iter__(self):
        for line in open('../data/mycorpus'):
            yield dictionary.doc2bow(line.lower().split())

corpusMemoryFriendly = MyCorpus()  # doesn't load the corpus into memory!
for vector in corpusMemoryFriendly:  # load one vector into memory at a time
    print(vector)