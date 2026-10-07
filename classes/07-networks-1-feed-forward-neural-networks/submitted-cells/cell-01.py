import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy #changed it from binary given this is not a binary task

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = x_train.astype('float32') / 255
y_train = to_categorical(y_train)

#added these two lines, we have to treat the training data the same as the validation
x_val = x_val.astype('float32') / 255
y_val = to_categorical(y_val)


model = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(10, activation='softmax') #we do not want to work with logits we want to work with probabilities
])

model.compile(optimizer='SGD',
              loss=CategoricalCrossentropy(), #change here
              metrics=['accuracy'])

res = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

print("results: ", res)
