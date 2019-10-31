import pandas as pd
from gensim.models.word2vec import Word2Vec


# def get_text(contents):
#     # contents以空格切分的句子列表
#     return [content.split() for content in contents]
#
# train_data = pd.read_csv(r'../dataset/train_char.csv')
# train_text = get_text(train_data.content)
#
# valid_data = pd.read_csv(r'../dataset/valid_char.csv')
# valid_text = get_text(valid_data.content)
#
# test_data = pd.read_csv(r'../dataset/test_char.csv')
# test_text = get_text(test_data.content)
#
# text = train_text + valid_text + test_text
#
# word2vec_model = Word2Vec(text, size=100, window=10, min_count=5, workers=4)
#
# word2vec_model.save(r'../dataset/w2v.model')
word2vec_model = Word2Vec.load(r'../dataset/w2v.model')
print(word2vec_model.wv['吃'])