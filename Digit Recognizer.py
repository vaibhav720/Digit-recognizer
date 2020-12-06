#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np # linear algebra
import pandas as pd
import os
import numpy as np
import glob
import shutil
import pandas as pd;
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from keras.layers.normalization import BatchNormalization


# In[17]:


train_full = pd.read_csv("E:/project/digit_recognizer/train.csv")
test= pd.read_csv("E:/project/digit_recognizer/test.csv")
train_full.head()


# In[18]:


train = train_full.sample(frac=0.8, random_state=0)
val = train_full.drop(train.index)
total_full = train_full.shape[0]
total_train = train.shape[0]
total_val = val.shape[0]
X_full = (train_full.iloc[:,1:].values).astype('float32') 
y_full = train_full.iloc[:,0].values.astype('int32') 

X_train = (train.iloc[:,1:].values).astype('float32') 
y_train = train.iloc[:,0].values.astype('int32') 

X_val = (val.iloc[:,1:].values).astype('float32') 
y_val = val.iloc[:,0].values.astype('int32') 

X_test = test.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0], 28, 28)

plt.figure(figsize=(10,10))
i = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()


# In[12]:


X_full = X_full.reshape(X_full.shape[0], 28, 28,1)
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_val = X_val.reshape(X_val.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

BATCH_SIZE = 100

image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      horizontal_flip=False,
      fill_mode='nearest')


train_data_gen = image_gen_train.flow(X_train, y_train, batch_size=BATCH_SIZE,
                                                     shuffle=True)

full_data_gen = image_gen_train.flow(X_full, y_full, batch_size=BATCH_SIZE,
                                                     shuffle=True)
#Normalization et validation batchs preparation
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow(X_val, y_val, batch_size=BATCH_SIZE)
#Display first 25 images of first batch
batchNo = 0
image_dim = train_data_gen[batchNo][0]
label_dim = train_data_gen[batchNo][1]
images_lot = image_dim.reshape(image_dim.shape[0], 28, 28)

plt.figure(figsize=(10,10))
i = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_lot[i], cmap=plt.cm.binary)
    plt.xlabel(label_dim[i])


# In[19]:


model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
   tf.keras.layers.BatchNormalization(axis=1),
   tf.keras.layers.Conv2D(32, (3,3), activation='relu'),     
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
   tf.keras.layers.BatchNormalization(axis=1),
   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),   
   tf.keras.layers.MaxPooling2D(2,2),
   tf.keras.layers.Dropout(0.2),  #DROPOUT 20% of neuron during process
   tf.keras.layers.Flatten(),

   tf.keras.layers.Dense(512, activation='relu'),
   tf.keras.layers.Dropout(0.2),  #DROPOUT 20% of neuron during process
   tf.keras.layers.Dense(10, activation='softmax')
])


# In[20]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# In[21]:


#Model training (on 20 epochs)
epochs=20
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)


# In[22]:


#Plot training and validation graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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


# In[23]:


#Model training on full train set (train + validation)
epochs=17
history = model.fit_generator(
    full_data_gen,
    steps_per_epoch=int(np.ceil(total_full / float(BATCH_SIZE))),
    epochs=epochs
)


# In[ ]:





# In[ ]:




