import jieba
import re
import pandas as pd
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

def cut(string): return ' '.join(jieba.cut(string))

def token(string):
    # 去掉标点，但是保留表情符号
    clean_str = re.sub(r"，|,|。|!|？|;|:|：|；|<|>|《|》|[|]|/|`|~|'|'|\"|\"|\|！|…|【|】","",string)
    clean_str = clean_str.strip().replace(' ', '').replace('\n', '')
    return clean_str

def del_stopwords(string):
    stop_words = get_stop_words()
    words = string.split()
    # words_del_stopwords = list(set(words).difference(set(stop_words))) 这种方法无序
    words_del_stopwords = [i for i in words if i not in stop_words]
    return ' '.join(words_del_stopwords)


def data_process(source_filepath, clean_data_filepath):
    comments = pd.read_csv(source_filepath)
    comments.dropna(inplace=True)
    comments.drop_duplicates(subset=['content'], inplace=True)
    content_cut = [cut(token(content)) for content in comments['content']]
    comments['content_cut'] = content_cut
    content_cut_del_stopwords = [del_stopwords(content) for content in content_cut]
    comments['content_cut_del_stopwords'] = content_cut_del_stopwords
    comments.dropna(subset=['content_cut_del_stopwords'], inplace=True)
    comments.to_csv(clean_data_filepath)

if __name__ == '__main__':
    data_process(
        r'../dataset/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv',
        r'../dataset/train.csv')
    data_process(r'../dataset/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv', r'../dataset/valid.csv')

