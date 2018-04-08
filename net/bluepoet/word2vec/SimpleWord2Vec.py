from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

with open('./test.txt', 'r', encoding= 'utf-8') as f:
    text = f.readlines()
token = [s.split() for s in text]

print(token)
embedding = word2vec.Word2Vec(token, size=1000, window=1, negative=3, min_count=1)
embedding.wv.save_word2vec_format('my.embedding', binary=False)

model = KeyedVectors.load_word2vec_format('my.embedding', binary=False, encoding='utf-8')

# print(model.wv['나는'])
print(model.most_similar('너를'))