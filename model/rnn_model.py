from gensim.models.KeyedVectors import load_word2vec_format

from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D


class RnnModel:
    model: Sequential = None
    w2v = None
    predicate_classes: int = 0

    def __init__(self, predicate_classes=5, w2v_path=None, w2v_bin=True):
        self.model = None
        self.w2v = load_word2vec_format(w2v_path, binary=True)
        self.predicate_classes = predicate_classes

    def create(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=len(self.w2v[0][0]),
                              kernel_size=3,
                              padding='same',
                              activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100))
        self.model.add(Dense(self.predicate_classes, activation='sigmoid'))
        self.model.compile(optimizer='adam')
