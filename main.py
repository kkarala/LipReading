import os
import numpy as np
import math
import keras
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



def step_decay(epoch): # reduce lr every 30 epochs
   initial_lrate = 1e-3
   drop = 0.5
   epochs_drop = 30.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

def build_model():
	model = Sequential()
	model.add(Convolution2D(32,(2,2),input_shape = (224,224,1), activation = 'relu',strides=2))
	model.add(Convolution2D(64,(3,3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Convolution2D(64,(3,3),activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Convolution2D(64,(3,3),activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Flatten())
	model.add(BatchNormalization(scale = False))
	model.add(Dropout((0.4)))
	model.add(Dense(1024, activation = 'relu'))
	model.add(BatchNormalization(scale = False))
	model.add(Dropout((0.5)))
	model.add(Dense(20, activation = 'softmax'))
	return model

model = build_model()

plot_model(model, show_layer_names=False, to_file='model.png')
model.summary()

lrate = LearningRateScheduler(step_decay)

model.compile(optimizer = SGD(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_folder = "miraclvc1_processed/train"
valid_folder = "miraclvc1_processed/development"
test_folder = "miraclvc1_processed/test"

train_set = train_datagen.flow_from_directory(train_folder,target_size=(224, 224),batch_size=32,class_mode='categorical', color_mode="grayscale",shuffle=True)

valid_set = valid_datagen.flow_from_directory(valid_folder,target_size=(224, 224),batch_size=1,class_mode='categorical', color_mode="grayscale")

test_set = test_datagen.flow_from_directory(test_folder,target_size=(224, 224),batch_size=1,class_mode='categorical', color_mode="grayscale")

STEP_SIZE_TRAIN=train_set.n//train_set.batch_size
STEP_SIZE_VALID=valid_set.n//valid_set.batch_size
STEP_SIZE_TEST=test_set.n//test_set.batch_size

history = model.fit_generator(train_set,steps_per_epoch=STEP_SIZE_TRAIN,epochs=100,validation_data=valid_set,validation_steps=STEP_SIZE_VALID,callbacks=[],verbose=2)



score = model.evaluate_generator(test_set,steps=STEP_SIZE_TEST,verbose=1)
print("TEST SCORE:", score)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc.png', bbox_inches='tight')
plt.close()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png', bbox_inches='tight')


