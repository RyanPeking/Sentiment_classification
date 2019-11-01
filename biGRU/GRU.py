import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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


train_path = '../dataset/train.csv'
valid_path = '../dataset/valid.csv'
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
    # train = pd.read_csv(train_path, lineterminator='\n')
    train = pd.read_csv(train_path)
    X_train = train['content_cut_del_stopwords']
    y_train = train[category]
    y_train = reformat(y_train)

    # valid = pd.read_csv(valid_path, lineterminator='\n')
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


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
#         val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        # print(' — val_f1:' ,_val_f1)





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


def train_model(i, category, category_num):
    print('training({}/{}): {}'.format(str(i+1), str(category_num), category))
    model = get_model()
    batch_size = 32
    epochs = 5
    X_train, y_train, X_valid, y_valid = get_data(train_path, valid_path, category)
    # RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)
    metrics = Metrics()
    class_weights = {
        0: 1,
        1: 5,
        2: 5,
        3: 0.5
    }

    model.fit(X_train, y_train, class_weight=class_weights, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),
                     callbacks=[metrics], verbose=2)
    print('-----------------------------------------------------------------------------------------------')



if __name__ == '__main__':
    categories = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
                  'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
                  'price_level', 'price_cost_effective', 'price_discount', 'environment_decoration', 'environment_noise',
                  'environment_space', 'environment_cleaness', 'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
                  'others_overall_experience', 'others_willing_to_consume_again']
    for i, category in enumerate(categories):
        train_model(i, category, len(categories))

