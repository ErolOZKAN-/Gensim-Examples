import gensim, logging, os
from gensim import models
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname == "mycorpus":
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.lower().split()

sentences = MySentences('../data/') # a memory-friendly iterator
for i in sentences:
    print(i)

model = models.Word2Vec(sentences,min_count=1)
print  model.wv.most_similar(positive=['human'])

model.save("../output/word2vecmodel")
model = Word2Vec.load("../output/word2vecmodel")