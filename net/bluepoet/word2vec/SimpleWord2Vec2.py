from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from konlpy.corpus import kobill
from konlpy.tag import Twitter

t = Twitter()
fields_ko = kobill.fileids()
docs_ko = kobill.open('1809890.txt').read()
tokens_ko = t.morphs(docs_ko)
print(isinstance(tokens_ko, list))
print(tokens_ko)

embedding = word2vec.Word2Vec(tokens_ko, size=5, window=1, negative=3, min_count=1)

# token으론 잘 짤리는 데 왜 한글자로 저장되는지 모르겠음
embedding.wv.save_word2vec_format('my.sample', binary=False)

model = KeyedVectors.load_word2vec_format('my.sample', binary=False, encoding='utf-8')

print(model.most_similar('육'))