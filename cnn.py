#%%
import tensorflow as tf
from tensorflow import keras
import os
import time

train = keras.preprocessing.image_dataset_from_directory("Plantvillage",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256,256),
    batch_size=32
)


val = keras.preprocessing.image_dataset_from_directory("Plantvillage",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256,256),
    batch_size=32
)

class_names = train.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE

train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val = val.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_train = train.map(lambda x,y: (normalization_layer(x),y))


model = keras.Sequential()
#model.add(data_augmentation)
model.add(keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(256, 256,3)))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(15))

# %%
print(class_names)
print(len(class_names))
# %%
model.compile(optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
#model.summary()

root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

checkpoint_cb = keras.callbacks.ModelCheckpoint("cnn2.h5", save_best_only=True)

model = keras.models.load_model("cnn2.h5")

# %%
epochs = 1
history = model.fit(
    normalized_train,
    validation_data=val,
    epochs=epochs,
    callbacks=[tensorboard_cb,checkpoint_cb]
)
# %%
# PARTIE VISUALISATION
modelImageVisualization = keras.Model(inputs=model.inputs, outputs=model.layers[2].output)
# %%
feature_maps = modelImageVisualization.predict(val.take(1))
# %%
import matplotlib.pyplot as plt
for i in range(4):
    ax = plt.subplot(2,2,i+1)
    plt.imshow(feature_maps[19,:,:,i],cmap="gray")
plt.show()

#%%
filters,biases=model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
for i in range(6):
    for j in range(3):
        ax = plt.subplot(6, 3, (j+1) + i * 3)
        plt.imshow(filters[:,:,j,i], cmap="gray")

plt.show()
# %%
