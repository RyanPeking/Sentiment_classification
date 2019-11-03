import pickle
from keras.preprocessing import sequence
from TextCNN.TextCNN import get_model


maxlen = 260
tokener_filepath = './dataset/tokenizer.pickle'

def load_tokener():
    with open(tokener_filepath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def data_process(comment):
    tokenizer = load_tokener()
    query = tokenizer.texts_to_sequences(comment)
    query = sequence.pad_sequences(query, maxlen=maxlen)
    return query

def predict(query):
    model = get_model()
    # model.load_weights('./TextCNN/textcnn_model_location_traffic_convenience_01.hdf5')
    print(model.predict(query))



if __name__ == '__main__':
    query = data_process(['第三次 参加 大众 点评 网 霸王餐 的 活动 这家 店 给 人 整体 感觉 一般 首先 环境 只能 算 中等 其次 霸王餐 提供 的 菜品 也 不是 很多 当然 商家 为了 避免 参加 霸王餐 吃不饱 的 现象 给 每桌 都 提供 了 至少 六份 主食 我们 那桌 都 提供 了 两份 年糕 第一次 吃火锅 会 在 桌上 有 这么 多 的 主食 了 整体 来说 这家 火锅店 没有 什么 特别 有 特色 的 不过 每份 菜品 分量 还是 比较 足 的 这点 要 肯定'])
    predict(query)

