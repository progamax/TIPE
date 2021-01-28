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
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])
image, result = next(train.take(1).as_numpy_iterator())
result = classifier.predict(image)

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


# %%
classe = []
for i in range(len(result)):
    classe.append(np.argmax(result[i], axis=-1))
    print(imagenet_labels[classe[i]])
# %%
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image7.jpg','https://i.ytimg.com/vi/qFpAZItPWUk/maxresdefault.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result)
predicted_class = np.argmax(result[0], axis=-1)
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

# %%
plt.imshow(image[5,:,:,:]/255)
# %%

plt.bar(imagenet_labels, result[0])
plt.show()
# %%
