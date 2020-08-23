#%%
import tensorflow as tf
from tensorflow import keras
import os
# %%
tf.__version__
# %%
train = keras.preprocessing.image_dataset_from_directory("PlantVillage/PlantVillage",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256,256),
    batch_size=32)

# %%
val = keras.preprocessing.image_dataset_from_directory("PlantVillage/PlantVillage",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256,256),
    batch_size=32)

# %%
class_names = train.class_names
# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE

train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val = val.cache().prefetch(buffer_size=AUTOTUNE)

# %%
normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)

# %%
normalized_train = train.map(lambda x,y: (normalization_layer(x),y))



model = keras.Sequential()
model.add(keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(256,256,3)))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(15))

model.compile(optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
model.summary()
#%%
import os
import time
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
# %%
epochs = 10
history = model.fit(
    normalized_train,
    validation_data=val,
    epochs=epochs,
    callbacks=[tensorboard_cb]
)
# %%
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# %%
