from numpy.random import random
from numpy import array, cumsum
from keras.models import Sequential
from keras.layers import LSTM, Dense, Distributed, Bidirectional


class BiLSTM:
    model: Sequential = None

    def __init__(self):
        self.model = None

    def create(self, output_classes):
        self.model = Sequential()
        self.model.add(Bidirectional(
            LSTM(output_classes), return_sequences=True),
            input_shape=None)
