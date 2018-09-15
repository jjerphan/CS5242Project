import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from keras.utils import to_categorical
np.random.seed(0)

data = load_iris()
nb_ex = 12
channels = 3
size = 32
X = np.random.random((nb_ex, channels, size, size))
Y = data.target
y_binary = to_categorical(Y)

print(X.shape)
print(Y.shape)
model = Sequential()

model.add(Dense(4, input_shape=(), activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(len(set(Y)), activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y_binary)
