import logging
from gensim import corpora
from collections import defaultdict
from pprint import pprint # pretty-printer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MINCOUNTTRESHOLD = 1
stopWordList = 'for a of the and to in'

def getData():
    return  ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

def getStopList(stopWordList): # return list of stop words given stop words string
    return set(stopWordList.split())

def parseDocuments(documents, stoplist): # parses documents and splits words
    return [[word for word in document.lower().split() if word not in stoplist]
            for document in documents]

def getVocabFrequency(texts): # creates vocab frequencies
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency

def removeNonFrequentWords(texts, frequency, MINCOUNTTRESHOLD): # removes words that below a given treshod
    # remove words that appear only once
    return [[token for token in text if frequency[token] > MINCOUNTTRESHOLD]
            for text in texts]

def convertToWordVector(dictionary, document):
    return dictionary.doc2bow(document.lower().split()) # converts a document to word vector format

def storeToDisk(corpus, fileName):
    corpora.MmCorpus.serialize(fileName, corpus)  # store to disk, for later use

def loadFromDisk(fileName):
    return corpora.MmCorpus(fileName) # load from disk

documents = getData()
stoplist = getStopList(stopWordList)
texts = parseDocuments(documents, stoplist)
frequency = getVocabFrequency(texts)
texts = removeNonFrequentWords(texts, frequency, MINCOUNTTRESHOLD)
print "PARSED TEXT : "
pprint(texts)

# CREATE DICTIONARY
dictionary = corpora.Dictionary(texts)
dictionary.save('../output/dictionary.dict')
print "DICTIONARY : "
print(dictionary)
print(dictionary.token2id)

# DEFINE NEW DOCUMENT AND CONVERT IT TO VECTOR SPACE MODEL
newDocument = "Human computer interaction"
newDocumentVector = convertToWordVector(dictionary, newDocument)
print "NEW DOCUMENT VECTOR SPACE MODEL : "
print(newDocumentVector)

#CREATE CORPUS
corpus = [dictionary.doc2bow(text) for text in texts]

#STORE CORPUS
fileName = '../output/corpus.mm'
print "STOREING CORPUS TO DISK : "
storeToDisk(corpus, fileName)
print(corpus)

#LOAD CORPUS
print "LOADING CORPUS FROM DISK : "
corpus = loadFromDisk(fileName)
print(list(corpus))



