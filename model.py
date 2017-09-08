import os
import csv

samples = []

with open('./mydata/mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

# A generator function that continuously generates batches of data for training the neural network.
#This will avoid loading all training data into the memory at once and helps saving memory.
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                center_name = './mydata/mydata/IMG/'+batch_sample[0].split('\\')[-1]
                left_name = './mydata/mydata/IMG/'+batch_sample[1].split('\\')[-1]
                right_name = './mydata/mydata/IMG/'+batch_sample[2].split('\\')[-1]
                
                # importing images
                # to include side camera images in the training, uncomment the following lines.
                center_image = cv2.imread(center_name)
                #left_image = cv2.imread(left_name)
                #right_image = cv2.imread(right_name)
                
                # reading the input steering angles for every image
                center_angle = float(batch_sample[3])
                #left_angle = center_angle+0.1
                #right_angle = center_angle-0.1                
            
                images.append(center_image)
                #images.append(left_image)
                #images.append(right_image)
                
                # flipping images to augment the data
                images.append(np.fliplr(center_image))
                #images.append(np.fliplr(left_image))
                #images.append(np.fliplr(right_image))
                
                angles.append(center_angle)
                #angles.append(left_angle)
                #angles.append(right_angle)

                angles.append(-1*center_angle)
                #angles.append(-1*left_angle)
                #angles.append(-1*right_angle)
               
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)





from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))

model.add(Conv2D(24,(5,5),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(36,(5,5),activation="relu"))
model.add(MaxPooling2D(2,2))
#model.add(Dropout(0.25))

model.add(Conv2D(48,(5,5),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(1,1))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(1,1))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(800,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')  
model.fit_generator(train_generator, validation_data=validation_generator,steps_per_epoch=200, epochs=5, validation_steps=20)
         
model.save('model.h5')
