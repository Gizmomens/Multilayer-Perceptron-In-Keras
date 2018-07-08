import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import csv
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from keras.utils import plot_model

train_samples = []
train_labels = []

no_of_features = 96
firstLine = True

with open('Training.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if firstLine:
            firstLine = False
        else:
            list = []
            for i in range(0, no_of_features):
                list.append(float(row[i]))

            train_samples.append(np.array(list))
            train_labels.append(row[no_of_features])


#encodes the labels 
encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(train_labels)


train_samples = np.array(train_samples)
train_labels = np.array(transfomed_label)

#print(train_samples)
#print(train_labels)

scaler = MinMaxScaler(feature_range=(-1,1))
scaled_train_samples = (scaler.fit_transform(train_samples))

#shuffle the rows/samples and the labels randomly 
# corresponding labels will still match!!! )
scaled_train_samples, train_labels = shuffle(scaled_train_samples, train_labels, random_state=0)


print(scaled_train_samples)
print(train_labels)

model = Sequential([
    Dense(64, input_shape=(96,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])


model.summary()


model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(scaled_train_samples, train_labels, validation_split = 0.1, batch_size=10, epochs=150, shuffle=True, verbose=2)
