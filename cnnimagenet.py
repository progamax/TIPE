#%%
import tensorflow as tf
from tensorflow import keras
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow_hub as hub
classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
#classifier_model = "https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1"
img_size = 224

train = keras.preprocessing.image_dataset_from_directory("Plantvillage_Relabelled",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size,img_size),
    batch_size=32
)


val = keras.preprocessing.image_dataset_from_directory("Plantvillage_Relabelled",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size,img_size),
    batch_size=32
)

class_names = train.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE

train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val = val.cache().prefetch(buffer_size=AUTOTUNE)
# %%
# PARTIE IMAGE NET
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    #hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
    keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size, img_size,3)),
    hub.KerasLayer(classifier_model),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(15)
])

classifier.compile(optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])


#%%
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


checkpoint_cb = keras.callbacks.ModelCheckpoint("cnn-imagenet2.h5", save_best_only=True)
#%%
history = classifier.fit(
    train,
    validation_data=val,
    epochs=20,
    callbacks=[tensorboard_cb,checkpoint_cb]
)
#%%




image, result = next(train.take(1).as_numpy_iterator())
result = classifier.predict(image)

#labels = np.array(open("plantsLabel.txt").read().splitlines())
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# %%
classe = []
for i in range(len(result)):
    classe.append(np.argmax(result[i], axis=-1))
    print(imagenet_labels[classe[i]])
# %%
import PIL.Image as Image

# %%
plt.imshow(image[5,:,:,:]/255)
# %%

plt.bar(labels, result[0])
plt.show()
# %%
