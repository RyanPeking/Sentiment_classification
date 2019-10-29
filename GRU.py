import pandas as pd
from sklearn.metrics import roc_auc_score
from config import train_path, valid_path
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import numpy as np

# 解决AlreadyExistsError的bug
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from keras.backend import set_session
tf.keras.backend.clear_session()  # For easy reset of notebook state.
config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.Session(config=config_proto)
set_session(session)



max_features = 50000
maxlen = 260  # 根据fix_length计算得出长度为260大概能覆盖98%的数据
embed_size = 300

def reformat(labels):
    y = []
    for label in labels:
        if label == 1:
            y.append([1, 0, 0, 0])
        if label == 0:
            y.append([0, 1, 0, 0])
        if label == -1:
            y.append([0, 0, 1, 0])
        if label == -2:
            y.append([0, 0, 0, 1])
    return np.array(y)

def tokenizer(X_train, X_valid):
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_valid))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_valid = tokenizer.texts_to_sequences(X_valid)
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)
    return X_train, X_valid


def get_data(train_path, valid_path, category):
    train = pd.read_csv(train_path)
    X_train = train['content_cut_del_stopwords']
    y_train = train[category]
    y_train = reformat(y_train)

    valid = pd.read_csv(valid_path)
    X_valid = valid['content_cut_del_stopwords']
    y_valid = valid[category]
    y_valid = reformat(y_valid)

    X_train, X_valid = tokenizer(X_train, X_valid)

    return X_train, y_train, X_valid, y_valid


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            # verbose = 0 为不在标准输出流输出日志信息
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(4, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    return model





if __name__ == '__main__':
    # X_train, y_train, X_valid, y_valid = get_data(train_path, valid_path, 'location_traffic_convenience')
    X_train, y_train, X_valid, y_valid = get_data(train_path, valid_path, 'location_distance_from_business_district')

    model = get_model()

    batch_size = 32
    epochs = 5

    RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),
                     callbacks=[RocAuc], verbose=2)