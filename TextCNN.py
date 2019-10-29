from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import Callback
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.metrics import roc_auc_score
import pandas as pd
from config import train_path, valid_path


max_features = 50000
maxlen = 260  # 根据fix_length计算得出长度为260大概能覆盖98%的数据
embed_size = 300
filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5


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
    inputs = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(input_dim=max_features, output_dim=embed_size, input_length=maxlen)(inputs)
    reshape = Reshape((maxlen, embed_size, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=4, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    # checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid = get_data(train_path, valid_path, 'location_traffic_convenience')
    # X_train, y_train, X_valid, y_valid = get_data(train_path, valid_path, 'location_distance_from_business_district')

    model = get_model()

    batch_size = 32
    epochs = 5

    RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),
                     callbacks=[RocAuc], verbose=2)