import jieba
import re
import pandas
from hanziconv import HanziConv


def toSimplified(string):
    return HanziConv.toSimplified(string)

def cut(string): return ' '.join(jieba.cut(string))

def token(string):
    return re.findall(r'[\d|\w]+', string)

def data_process(source_filepath, clean_data_filepath):
