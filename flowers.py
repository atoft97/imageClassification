import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import os
from keras.utils import to_categorical

train_dir = os.path.join("flowers-recognition", 'train')
test_dir = os.path.join("flowers-recognition", 'test')
val_dir = os.path.join("flowers-recognition", 'validation')

train_daisy = os.path.join(train_dir, 'daisy')
train_dandelion = os.path.join(train_dir, 'dandelion')
train_rose = os.path.join(train_dir, 'rose')
train_sunflower = os.path.join(train_dir, 'sunflower')
train_tulip = os.path.join(train_dir, 'tulip')


test_daisy = os.path.join(test_dir, 'daisy')
test_dandelion = os.path.join(test_dir, 'dandelion')
test_rose = os.path.join(test_dir, 'rose')
test_sunflower = os.path.join(test_dir, 'sunflower')
test_tulip = os.path.join(test_dir, 'tulip')


num_train_daisy = len(os.listdir(train_daisy))
num_train_dandelion = len(os.listdir(train_dandelion))
num_train_rose = len(os.listdir(train_rose))
num_train_sunflower = len(os.listdir(train_sunflower))
num_train_tulip = len(os.listdir(train_tulip))

num_test_daisy = len(os.listdir(test_daisy))
num_test_dandelion = len(os.listdir(test_dandelion))
num_test_rose = len(os.listdir(test_rose))
num_test_sunflower = len(os.listdir(test_sunflower))
num_test_tulip = len(os.listdir(test_tulip))


total_train = num_train_daisy +num_train_dandelion +num_train_rose +num_train_sunflower + num_train_tulip
total_test = num_test_daisy + num_test_dandelion +num_test_rose +num_test_sunflower + num_test_tulip


train_image_generator = ImageDataGenerator(rescale=1./255,
                                          horizontal_flip=True,
                                          rotation_range=20,
                                          zoom_range=0.5,
                                          width_shift_range=.15,
                                          height_shift_range=.15,

                                          featurewise_center=False, 
                                          samplewise_center=False,
                                          featurewise_std_normalization=False, 
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          vertical_flip=False
                                          )

test_image_generator = ImageDataGenerator(rescale=1./255)

batch_size = 32
epochs = 1
IMG_HEIGHT = 150
IMG_WIDTH = 150

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
label_map = (train_data_gen.class_indices)
#print(label_map)

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5, activation="softmax")
])

model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=test_data_gen,
    validation_steps=total_test // batch_size
)



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


def plotResult(images_arr, predicted):
    fig, axes = plt.subplots(2, 3, figsize=(10,15))
    axes = axes.flatten()
    i = 0
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(predicted[i])
        i+=1
    plt.tight_layout()
    plt.show()

test_image_generator2 = ImageDataGenerator(rescale=1./255)

def getTitels(teller, predicted, correct):
  listOfTitels = []
  #j = 0
  #for liste in predicted:
  k = teller + 6
  if (k >len(predicted)):
    k = len(predicted)

  for i in range(teller, k):
    liste = predicted[i]
    #print(liste)
    tekst = ""
    for x in range(len(liste)):
      liste[x] = round(liste[x] * 100, 2)
      tekst += str(listOfLables[x]) +  ": " + str(liste[x]) + "%  \n "
    print(i)
    tekst += "Korrekt: " + correct[i] 
    listOfTitels.append(tekst)
    #j+=1
  return(listOfTitels)

listOfLables = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

##############################################################################

correct3 = ["Daisy", "Dandelion","Dandelion",  "Rose", "Sunflower", "Tulip"]

test_data_gen3 = test_image_generator2.flow_from_directory(batch_size=6,
                                                           directory=val_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary',
                                                           shuffle=False)

predicted3 = model.predict(test_data_gen3)

sampled3, _ = next(test_data_gen3, -1)

listOfTitels3 = []
j = 0
for liste in predicted3:
  print(liste)
  tekst = ""
  for i in range(len(liste)):
    liste[i] = round(liste[i] * 100, 2)
    tekst += str(listOfLables[i]) +  ": " + str(liste[i]) + "%  \n "
  tekst += "Korrekt: " + correct3[j] 
  listOfTitels3.append(tekst)
  j+=1


#titels3 = getTitels(6, predicted3)

print(listOfTitels3)

plotResult(sampled3, listOfTitels3)

###############################################################################








test_data_gen2 = test_image_generator2.flow_from_directory(batch_size=6,
                                                           directory=test_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary',
                                                           shuffle=False)

filenames = test_data_gen2.filenames
nb_samples = len(filenames)

test_data_gen2.reset()

#predicted = model.predict(test_data_gen2, steps = 1)
predicted = model.predict(test_data_gen2)

print(predicted)



correct = []

for i in range (49):
  correct.append("Daisy")
for i in range (49):
  correct.append("Dandelion")
for i in range (42):
  correct.append("Rose")
for i in range (42):
  correct.append("Sunflower")
for i in range (49):
  correct.append("Tulip")





print(correct)

#print(listOfTitels)


i = 0
while (True):
  sampled, _ = next(test_data_gen2, -1)
  if (i > len(predicted)):
    break
  titels = getTitels(i, predicted, correct)
  print(titels)
  plotResult(sampled, titels)
  i += 6


#sampled, _ = next(test_data_gen2, -1)


#plotResult(sampled, listOfTitels)