
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import theano


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import csv
import time

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

#-----------------------------------------------------------------------------
def flip(image,steering):

    flipped_image = cv2.flip(image, 1) #flip around the vertical axes
    steer_value =  steering * -1.0  #flip the steering angles

    return flipped_image, steer_value


def augment_brightness_camera_images(image):

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#-----------------------------------------------------------------------------

#hints from the forum:
#thanks to the posts of @subodh.malgonde & @brenkenobi1990
#https://discussions.udacity.com/t/using-generator-to-implement-random-augmentations/242185/9

#-----------------------------------------------------------------------------

def my_generator(lines, batch_size=32):

    N = (len(lines)// batch_size)*batch_size  # make the number of samples in 'lines' a multiple of batch_size

    correction = 0.16

    while True:
        for i in range(0, N, batch_size):

            images = []
            steer_values = []

            batch_start = i
            batch_stop = i + batch_size
            batch_lines = lines[batch_start:batch_stop]

            for j, line in enumerate(batch_lines):

                for i in range(3): # considering the input of all three cameras

                    source_path = line[i] # 0-center image, 1-left image, 2-right image
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = './data/IMG/'+filename

                    image = mpimg.imread(local_path)

                    # crop
                    image = image[50:135,:,:]  #50-135

                    # resize
                    image = cv2.resize(image, (64, 64))

                    #augment some addtional data

                    noisy_img = augment_brightness_camera_images(image)

                    steer_reading = float(line[3])  #steering


                    if(i==1):
                        steer_reading = steer_reading + correction # adjust left image from the center

                    elif(i==2):
                        steer_reading = steer_reading - correction # adjust right image from the center


                    flipped_img, flipped_steer = flip(image,steer_reading)

                    images.append(image)
                    images.append(noisy_img)
                    images.append(flipped_img)

                    steer_values.append(steer_reading)
                    steer_values.append(steer_reading) # for noisy image
                    steer_values.append(flipped_steer)


            yield (np.array(images), np.array(steer_values))

#-----------------------------------------------------------


from sklearn.model_selection import train_test_split
lines = []

with open('data/driving_log.csv') as file:
    reader = csv.reader(file)
    next(reader, None)  # skip the headers

    for line in reader:
        lines.append(line)


line_sub = lines[:41000]
#line_sub = lines[:200]

train_lines, val_lines = train_test_split(line_sub, test_size=0.2)

training_generator = my_generator(train_lines)
valid_generator = my_generator(val_lines)

#-----------------------------------------------------------------------------
model = Sequential()

#normalize data: [-0.5-0.5]
#model.add(Lambda(lambda x:x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Lambda(lambda x:x/255.0 -0.5, input_shape=(64,64,3)))

#Perform some croping
#model.add(Cropping2D(cropping=((70,25),(0,0)))) #do not crop from left or right

model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(.25))
model.add(Convolution2D(48 ,(5,5), strides=(2,2), activation='relu'))
model.add(Convolution2D(64 ,(3,3), activation='relu'))
model.add(Convolution2D(64 ,(3,3), activation='relu'))
model.add(Dropout(.1))


model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))



model.add(Dense(1)) #regression task
#-----------------------------------------------------------------------------

t=time.time()

model.compile(optimizer='adam', loss='mse')
#model.fit(np.array(images), np.array(measurements), validation_split=0.2, shuffle=True, epochs=2)

model.fit_generator(training_generator, steps_per_epoch= len(train_lines),
                        validation_data=valid_generator, nb_val_samples=len(val_lines), nb_epoch=4)


model.save('save_model.h5')

t2 = time.time()

print()
print(round((t2-t)/60.0, 2), 'Minutes to Craete & Save Simple Keras Model for 500 Sample Data...')
