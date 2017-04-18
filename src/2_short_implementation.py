from six import iteritems
import logging
from gensim import corpora
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stopWordList = 'for a of the and to in'

def getStopList(stopWordList):
    return set(stopWordList.split())

#get stoplist
stoplist = getStopList(stopWordList)

# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('../data/mycorpus'))

# remove stop words
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]

# remove words that appear only once
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]

# apply filter to remove stop words and words that appear only once
dictionary.filter_tokens(stop_ids + once_ids)

# remove gaps in id sequence after words that were removed
dictionary.compactify()

#fucking print the dictinary
print(dictionary)

#save it
dictionary.save('../output/dictionary_v2.dict')
