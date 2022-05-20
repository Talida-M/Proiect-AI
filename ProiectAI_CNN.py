###CEA MAI BUNA SUBMISIE KAGGLE: 0.65340###

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as geek
import ipyplot
import random
from skimage import transform
from skimage.transform import rotate, AffineTransform,warp
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
from sklearn.metrics import confusion_matrix
import random
from skimage import img_as_ubyte
from skimage import exposure
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


#deschidem fisierele .txt si cream vectorii corespunzatori apoi ii convertim la numpy array
sample = []
f = open("sample_submission.txt", "r")
for i in f.readlines():
    sample.append(i.replace('\n', ''))
f.close()
np.array(sample)
test = []
f = open("test.txt", "r")  
for i in f.readlines():
    test.append(i.replace('\n', ''))
f.close()
test.remove('id')
test = np.array(test)
train = []
t = 0
f = open("train.txt", "r")
for i in f.readlines():
    ls = []
    for j in i.split(','):
        #print(j)
        ls.append(j.replace('\n', ''))
    train.append(ls)
    if train[t][1] != 'label':
        train[t][1] = float(train[t][1])
    t += 1
f.close()
train.remove(train[0])
train = np.array(train)
validation = []
v = 0
f = open("validation.txt", "r")
for i in f.readlines():
    ls = []
    for j in i.split(','):
        #print(j)
        ls.append(j.replace('\n', ''))
    validation.append(ls)
    if validation[v][1] != 'label':
        validation[v][1] = float(validation[v][1])
f.close()
validation.remove(validation[0])
validation = np.array(validation)

print(type(test))
#print(test)
print(validation)
print((train))

#deschidem imaginile, le punem in vectorii de tip numpy array corespunzatori si cream pentru train 
#si validation listele de categorii
label = []
label_t = []
label_v = []
m = 0
imagini_train = []
imagini_validation = []
imagini_test = []
for i in range(len(train)):
    m += 1
    imagini_train.append(np.array(Image.open("./train+validation/" + train[i][0])))
    label_t.append(int(float(train[i][1])))
imagini_train = np.array(imagini_train)    
for i in range(len(validation)):
    imagini_validation.append(np.array(Image.open("./train+validation/" + validation[i][0])))

    if validation[i][1] not in label:
        label.append(int(validation[i][1]))
    label_v.append(int(validation[i][1]))
imagini_validation = np.array(imagini_validation)      
for i in range(len(test)):
    imagini_test.append(np.array(Image.open("./test/" + test[i])))

label.sort()

imagini_test = np.array(imagini_test)
imagini_train.shape

imagini_validation.shape
print(imagini_test.shape)
print(imagini_train.shape)


#convertim labelurile la valori binare(in functie de pozitia categoriei)
label_convertit = to_categorical(label)
labelV_convertit = np.array(to_categorical(label_v))
labelT_convertit = np.array(to_categorical(label_t))
print(labelV_convertit)

#standardizam imaginile
m = preprocessing.MinMaxScaler()
imagini_test = imagini_test.reshape(-1,3*16*16)
imagini_validation = imagini_validation.reshape(-1,3*16*16)
imagini_train = imagini_train.reshape(-1,3*16*16)

scaler = StandardScaler()
imagini_test = scaler.fit_transform(imagini_test)
imagini_train = scaler.fit_transform(imagini_train)
imagini_validation = scaler.fit_transform(imagini_validation)

imagini_test = imagini_test.reshape(-1,16,16,3)
imagini_validation = imagini_validation.reshape(-1,16,16,3)
imagini_train = imagini_train.reshape(-1,16,16,3)
#normalizam pixelii pentru a putea corespunde cu label_convertit (valori intre [0,1])
imagini_test = cv2.normalize(imagini_test, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
imagini_validation = cv2.normalize(imagini_validation, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
imagini_train =  cv2.normalize(imagini_train, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print(imagini_validation)

#vom crea reteaua neuronala CNN - asemanatoare cu arhitectura AlexNet
print(len(imagini_train))
print(len(label_t))
from keras.layers import Dense, Activation
# input_shape = (16, 16, 3)
retea = Sequential()
retea.add(Conv2D(32, (3, 3), padding='same',  input_shape = (16,16, 3)))
retea.add(Activation('tanh'))
# retea.add(MaxPooling2D(pool_size = (2, 2)))
retea.add(Conv2D(32, (3, 3), padding='same'))
retea.add(Activation('tanh'))
# retea.add(MaxPooling2D(pool_size = (2, 2)))
retea.add(Conv2D(64, (3, 3),  padding='same'))
retea.add(Activation('tanh'))
retea.add(MaxPooling2D(pool_size = (2, 2)))
# retea.add(Flatten())
# retea.add(Dense(256, activation ='tanh'))
retea.add(Dropout(0.5))
retea.add(Conv2D(64, (3, 3),  padding='same'))
retea.add(Activation('tanh'))
retea.add(Conv2D(128, (3, 3), padding='same'))
retea.add(Activation('tanh'))
retea.add(Conv2D(256, (3, 3), padding='same'))
retea.add(Activation('tanh'))
retea.add(BatchNormalization())
retea.add(MaxPooling2D(pool_size = (2, 2)))
#retea.add(Flatten())
retea.add(Dropout(0.25))

retea.add(Conv2D(128, (3, 3), padding='same'))
retea.add(Activation('tanh'))
# retea.add(Dense(256, activation ='tanh'))
#retea.add(Activation('tanh'))
retea.add(BatchNormalization())
retea.add(MaxPooling2D(pool_size = (2, 2)))
retea.add(Dropout(0.5))

retea.add(Flatten())
retea.add(Dense(1000))
retea.add(Activation('tanh'))
retea.add(Dropout(0.5))
retea.add(Dense(7, activation='softmax'))

# retea.load_weights("sweights.h5")
# retea.save_weights('weights.h5')

retea.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# antrenam modelul de trei ori consecutiv
antrenare = retea.fit(imagini_train, labelT_convertit,
                      epochs = 10,
                      validation_data=(imagini_validation, labelV_convertit),
                      batch_size=512)


antrenare = retea.fit(imagini_train, labelT_convertit,
                      epochs = 30,
                      validation_data=(imagini_validation, labelV_convertit),
                      batch_size=128)


antrenare = retea.fit(imagini_train, labelT_convertit,
                      epochs = 10,
                      validation_data=(imagini_validation, labelV_convertit),
                      batch_size=512)

label_p  = retea.predict(imagini_validation)
label_p = np.argmax(label_p, axis = 1)

#facedm predictiile si afisam rezultatul final
pierdere, acuratete = retea.evaluate(imagini_validation, labelV_convertit)
labelt  = retea.predict(imagini_test)
labelv  = retea.predict(imagini_validation)
labelt = np.argmax(labelt, axis = 1)
labelv = np.argmax(labelv, axis = 1)
labelv_conv = np.argmax(labelV_convertit, axis = 1)
matrice_confuzie = confusion_matrix(labelv_conv, labelv)
label_test = np.array(to_categorical(labelt))
pierdere1, acuratete1 = retea.evaluate(imagini_test, label_test)
print(pierdere1, acuratete1)
print("id,label")
for i in range(len(imagini_test)):
    print(test[i]+ ',' + str(labelt[i]) )

#cream graficele necesare pentru date(utile la raport)
plt.plot(antrenare.history['accuracy'])
plt.plot(antrenare.history['val_accuracy'])
plt.title('Acuratetea Retelei Create')
plt.legend(['Antrenare', 'Validare'], loc='upper left')
plt.show()

plt.plot(antrenare.history['loss'])
plt.plot(antrenare.history['val_loss'])
plt.title('Loss-ul Retelei Create')
plt.legend(['Antrenare', 'Validare'], loc='upper left')
plt.show()

print(matrice_confuzie)