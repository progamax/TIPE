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

data_augmentation = tf.keras.Sequential([
  keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


model = keras.Sequential()
model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size, img_size,3)))
model.add(data_augmentation)
model.add(keras.layers.Conv2D(16//2, 3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(32//2, 3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(64//2, 3, padding="same", activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128/2, activation="relu"))
model.add(keras.layers.Dense(2))


# %%
print(class_names)
print(len(class_names))
# %%
model.compile(optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])
#model.summary()

modelImageVisualization1= keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
modelImageVisualization2 = keras.Model(inputs=model.inputs, outputs=model.layers[5].output)
modelImageVisualization3 = keras.Model(inputs=model.inputs, outputs=model.layers[7].output)

root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

file_writer = tf.summary.create_file_writer(run_logdir + '/cm')

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def image_callback(epoch, logs):
    image_batch, _ = next(iter(val))
    first_image = image_batch[0:1]
    result1 = modelImageVisualization1.predict(first_image)
    result2 = modelImageVisualization2.predict(first_image)
    result3 = modelImageVisualization3.predict(first_image)
    #result_tensor = tf.convert_to_tensor(result)
    #result_tensor = tf.expand_dims(result_tensor, axis=3)

    with file_writer.as_default():
        figure = plt.figure(figsize=(20,10))

        for i in range(32//2):
            plt.subplot(4,8,i+1)
            plt.imshow(result2[0,:,:,i])
        
        #tf.summary.image("Second Convolution Output - Card " + str(i+1), result_tensor[:,:,:,i:i+1], step=epoch)
        tf.summary.image("Convolution Output - Layer 2", plot_to_image(figure), step=epoch)

    with file_writer.as_default():
        figure = plt.figure(figsize=(20,10))

        for i in range(16//2):
            plt.subplot(4,4,i+1)
            plt.imshow(result1[0,:,:,i])
        
        #tf.summary.image("Second Convolution Output - Card " + str(i+1), result_tensor[:,:,:,i:i+1], step=epoch)
        tf.summary.image("Convolution Output - Layer 1", plot_to_image(figure), step=epoch)
    with file_writer.as_default():
        figure = plt.figure(figsize=(20,13))

        for i in range(64//2):
            plt.subplot(7,10,i+1)
            plt.imshow(result3[0,:,:,i])
        
        #tf.summary.image("Second Convolution Output - Card " + str(i+1), result_tensor[:,:,:,i:i+1], step=epoch)
        tf.summary.image("Convolution Output - Layer 3", plot_to_image(figure), step=epoch)

image_callback = keras.callbacks.LambdaCallback(on_epoch_end=image_callback)
#%%
checkpoint_cb = keras.callbacks.ModelCheckpoint("cnn9-label-imgsizenull.h5", save_best_only=True)
#model = keras.models.load_model("cnn2.h5")

# %%
epochs = 60
history = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    callbacks=[tensorboard_cb,checkpoint_cb,image_callback]
)

# %%



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
