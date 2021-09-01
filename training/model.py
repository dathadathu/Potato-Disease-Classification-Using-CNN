#loading required packages

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import experimental
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras import metrics
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import Resizing
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_logging_ops import Print
from tensorflow.python.ops.gen_math_ops import imag
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3

EPOCHS = 50


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Data",
    shuffle = True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE 
)

class_names = dataset.class_names

class_names

len(dataset)

plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.axis("off")
    plt.title(class_names[label_batch[i]])


def get_dataset_partitions_tf(ds, train_split = 0.8, test_split = 0.1, val_split = 0.1, shuffle = True, shuffle_size = 10000):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)

    train_size = int(ds_size*train_split)
    val_size = int(ds_size*val_split)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, test_ds, val_ds 

train_ds, test_ds, val_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_agumentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

inpt_shape = (IMAGE_SIZE, IMAGE_SIZE, BATCH_SIZE, CHANNEL_SIZE)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_agumentation,
    layers.Conv2D(32,(3,3), activation='relu',input_shape = inpt_shape),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,kernel_size= (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,kernel_size= (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,kernel_size= (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,kernel_size= (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,kernel_size= (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.build(input_shape=inpt_shape)

model.summary()

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

scores = model.evaluate(test_ds)

scores

print(history.params)

history.history.keys()
history.history.values()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label = "Training Accuracy")
plt.plot(range(EPOCHS), val_acc, label = "Validation Accuracy")
plt.title("Training and Validation accuracy")

plt.figure(figsize=(10,10))
plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label = "Training loss")
plt.plot(range(EPOCHS), val_loss, label = "Validation loss")
plt.title("Training and Validation loss")

for images_batch, labels_batch in test_ds.take(1):

    first_image = images_batch[0].numpy().astype('uint8')
    first_label = label_batch[0].numpy()

    plt.imshow(first_image)
    print("Actual Label:", class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("Predicted Label:", class_names[np.argmax(batch_prediction[0])])

