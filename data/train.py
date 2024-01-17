
# ## Coral
# %%
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as callbacks
from tensorflow import one_hot
import numpy as np
import os
os.chdir("/home/data")

# %%
(X_train,Y_train), (X_test,Y_test) = fashion_mnist.load_data()
X_train = tf.reshape(X_train, (-1, 28, 28, 1))
X_test = tf.reshape(X_test, (-1, 28, 28, 1))


# %%
es = callbacks.EarlyStopping(monitor='loss', patience=3)

# %%
model = Sequential()
model.add(layers.BatchNormalization(input_shape=X_train[0].shape))
model.add(layers.Conv2D(100, kernel_size=10))
model.add(layers.AveragePooling2D())
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(100, kernel_size=5))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(100))
model.add(layers.Dense(100))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.build()
model.summary()

# %%
hist = model.fit(X_train, one_hot(Y_train,10).numpy(), batch_size=64, validation_split=0.2, callbacks=[es], epochs = 3)


# %%
model.save("model.h5")
tf.saved_model.save(model, 'model')

# %%
np.savetxt("X_test.txt", X_test.numpy().reshape((10000,28*28)))

# %%
np.savetxt("Y_test.txt", Y_test)

# %%
classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
np.savetxt("classes.txt",classes, fmt="%s")


# %%
saved_keras_model = 'model.h5'
model.save(saved_keras_model)

def representative_dataset():
  for data in tf.data.Dataset.from_tensor_slices((X_test.numpy().reshape(-1,28,28,1))).batch(1).take(100):
    yield [tf.dtypes.cast(data, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8] # extra line missing
converter.representative_dataset=representative_dataset
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()


with open('model.tflite', 'wb') as f:
  f.write(tflite_model)