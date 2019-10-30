import random
random.seed = 16
import pandas as pd
from gensim.models.word2vec import Word2Vec
from langconv import *


stopwords_path = '../dataset/stop_words.txt'
# 加载停用词
def get_stop_words():
    stop_words = []
    for line in open(stopwords_path, 'r', encoding='utf-8'):
        stop_words.append(line.replace('\n', ''))
    return stop_words

def toSimplified(string):
    # 繁体转简体
    return Converter('zh-hans').convert(string)


# move stop words and generate char sent
def filter_char_map(arr):
    res = []
    stopwords = get_stop_words()
    for c in arr:
        if c not in stopwords and c != ' ' and c != '\xa0' and c != '\n' and c != '\ufeff' and c != '\r':
            res.append(c)
    return " ".join(res)


# get char of sentence
def get_char(arr):
    res = []
    for c in arr:
        res.append(c)
    return list(res)

def data_process(source_filepath, clean_data_filepath):
    comments = pd.read_csv(source_filepath)
    comments.dropna(inplace=True)
    comments.content = comments.content.map(lambda x: toSimplified(x))
    comments.content = comments.content.map(lambda x: filter_char_map(x))
    comments.content = comments.content.map(lambda x: get_char(x))
    comments.to_csv(clean_data_filepath, index=None)
    return comments.content

def word2vec(content, word2vec_filepath):
    line_sent = []
    for s in content:
        line_sent.append(s)
    word2vec_model = Word2Vec(line_sent, size=100, window=10, min_count=1, workers=4, iter=15)
    word2vec_model.wv.save_word2vec_format(word2vec_filepath, binary=True)

if __name__ == '__main__':
    train_content = data_process(
        r'../dataset/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv',
        r'../dataset/train_char.csv')
    valid_content = data_process(
        r'../dataset/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv',
        r'../dataset/valid_char.csv')
    test_content = data_process(
        r'../dataset/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv',
        r'../dataset/test_char.csv')
    word2vec(train_content + valid_content + test_content, r'../dataset/chars.vector.csv')

# line_sent = []
# for s in data["content"]:
#     line_sent.append(s)
# word2vec_model = Word2Vec(line_sent, size=100, window=10, min_count=1, workers=4, iter=15)
# word2vec_model.wv.save_word2vec_format("word2vec/chars.vector", binary=True)
#
# validation = pd.read_csv("ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")
# validation.content = validation.content.map(lambda x: filter_char_map(x))
# validation.content = validation.content.map(lambda x: get_char(x))
# validation.to_csv("preprocess/validation_char.csv", index=None)
#
# test = pd.read_csv("ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv")
# test.content = test.content.map(lambda x: filter_char_map(x))
# test.content = test.content.map(lambda x: get_char(x))
# test.to_csv("preprocess/test_char.csv", index=None)