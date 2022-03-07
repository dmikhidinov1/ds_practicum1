#!/usr/bin/env python
# coding: utf-8

# # MSDS692_X41_Data Science Practicum 
# ## Dilyor Mikhidinov
# Using Convolutional Neural Networks to classify Malaria Cell Images.

# In[1]:


#importing neccessary libraries:
import numpy as np 
import pandas as pd
import os
import time

#ML libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, InputLayer, Reshape, Conv1D, MaxPool1D, SeparableConv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_validate, train_test_split

import os
print(os.listdir("dataset/cell_images")) #folder where the dataset is located


# In[2]:


#Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# #### Checking if all the libraries are working properly. I have had very big issues with installing tensorflow_gpu and openCV

# In[3]:


import tensorflow as tf
print(tf.__version__)


# In[4]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[10]:


pip install opencv_python-4.5.5-cp38-cp38-win_amd64.whl


# In[5]:


import cv2


# ## Data Preprocessing

# Next I'm going to create seperate directory to manage labelled data. For fulfilling this task I am using shutil(https://docs.python.org/3/library/shutil.html) library. It allows variety of high-level operations on files (collection, copying and removal of files). 

# In[6]:


import shutil

base_dir = 'dataset/cell_images'
work_dir  = 'work/'
os.mkdir(work_dir)

base_dir_Positive = 'dataset/cell_images/Parasitized/' 
base_dir_Negative = 'dataset/cell_images/Uninfected/'

work_dir_Positive = 'work/Positive/'
os.mkdir(work_dir_Positive)
work_dir_Negative = 'work/Negative/'
os.mkdir(work_dir_Negative)


# Now we have work directory. Time to create training, validation and test folders with neg/pos (folder for negative and positive labelled images) inside each.

# In[7]:


train_dir = os.path.join(work_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(work_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(work_dir, 'test')
os.mkdir(test_dir)

print("New directories for train, validation, and test created")
train_pos_dir = os.path.join(train_dir, 'pos')
os.mkdir(train_pos_dir)
train_neg_dir = os.path.join(train_dir, 'neg')
os.mkdir(train_neg_dir)

validation_pos_dir = os.path.join(validation_dir, 'pos')
os.mkdir(validation_pos_dir)
validation_neg_dir = os.path.join(validation_dir, 'neg')
os.mkdir(validation_neg_dir)

test_pos_dir = os.path.join(test_dir, 'pos')
os.mkdir(test_pos_dir)
test_neg_dir = os.path.join(test_dir, 'neg')
os.mkdir(test_neg_dir)

print("Train, Validation, and Test folders made for both Positive and Negative datasets")


# In[8]:


i = 0
      
for filename in os.listdir(base_dir_Positive): 
    dst ="pos" + str(i) + ".jpg"
    src =base_dir_Positive + filename 
    dst =work_dir_Positive + dst 
          
       # rename() function will 
       # rename all the files 
    shutil.copy(src, dst) 
    i += 1


j = 0

for filename in os.listdir(base_dir_Negative): 
    dst ="neg" + str(j) + ".jpg"
    src =base_dir_Negative + filename 
    dst =work_dir_Negative + dst 
          
    # rename() function will 
    # rename all the files 
    shutil.copy(src, dst) 
    j += 1       
        
print("Images for both categories have been copied to working directories, renamed to <pos & neg + index> ")


# #### Next I am manually splitting all "Positive" labelled images into training, test and validation folders in 80%-10%-10% respectively

# In[9]:


fnames = ['pos{}.jpg'.format(i) for i in range(11000)]
for fname in fnames:
    src = os.path.join(work_dir_Positive, fname)
    dst = os.path.join(train_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(11000, 12390)]
for fname in fnames:
    src = os.path.join(work_dir_Positive, fname)
    dst = os.path.join(validation_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(12390, 13780)]
for fname in fnames:
    src = os.path.join(work_dir_Positive, fname)
    dst = os.path.join(test_pos_dir, fname)
    shutil.copyfile(src, dst)


# #### Next doing the same thing for "Negative" labelled images

# In[10]:


fnames = ['neg{}.jpg'.format(i) for i in range(11000)]
for fname in fnames:
    src = os.path.join(work_dir_Negative, fname)
    dst = os.path.join(train_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(11000, 12390)]
for fname in fnames:
    src = os.path.join(work_dir_Negative, fname)
    dst = os.path.join(validation_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(12390, 13780)]
for fname in fnames:
    src = os.path.join(work_dir_Negative, fname)
    dst = os.path.join(test_neg_dir, fname)
    shutil.copyfile(src, dst)


# #### Lets view distribution of images in each folder we created

# In[11]:


print("Train, validation, and test datasets split and ready for use")
print('total training pos images:', len(os.listdir(train_pos_dir)))
print('total training neg images:', len(os.listdir(train_neg_dir)))
print('total validation pos images:', len(os.listdir(validation_pos_dir)))
print('total validation neg images:', len(os.listdir(validation_neg_dir)))
print('total test pos images:', len(os.listdir(test_pos_dir)))
print('total test neg images:', len(os.listdir(test_neg_dir)))


# ## Image Augmentation

# Next I am going to use one of the powerful tools of tensorflow.keras called "ImageDataGenerator". This function allows us to take the path to a directory and generate batches of augmented data. Augmentation is important step in almost every Deep learning analysis. Since it allows to modify the existing data we are using in multiple manners so that the trained algorithm becomes capable of generating patterns for even more variety of images. The only case that Augmentation might not be applicable is when the goal is for example to predict the road signs for self driving cars. Signs are always fixed and do not appear for example in vertically flipped way in reality

# In[111]:


datagen = ImageDataGenerator(rescale=1./255)


# In[112]:


train_generator = datagen.flow_from_directory(directory = train_dir,
                                                    target_size=(128, 128),
                                                    class_mode='binary',
                                                    #subset='training',
                                                    shuffle=True,
                                                    batch_size=32)

valid_generator = datagen.flow_from_directory(directory = validation_dir,
                                                   target_size=(128, 128),
                                                   class_mode='binary',
                                                   #subset='validation',
                                                   shuffle = True,
                                                   batch_size=32)


classes = ['Parasitized', 'Uninfected']


# ## Some Visualizations

# In[18]:


sample_training_images, train_label = next(train_generator)


# In[130]:


def displayImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout() 
    plt.show()
print('Random Display of Cell images from Training set')
displayImages(sample_training_images[:5])


# In[ ]:


for i in range(5):
    img=cv2.imread(work_dir_Positive[i])
    plt.imshow(img)
    plt.title("Parasitized")
    plt.show()


# ## Looking at different CNN models

# During the project timeline I viewed different CNN models and checked their performances with evaluation metrics. And from the analysis I selected top 3 best performing models: DS-CNN, VGG19-InceptionV3

# #### Depth-Wise Separable CNN (DS-CNN)

# In[14]:


input_length = 128,128,3

ds_model = Sequential()
ds_model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.2))

ds_model.add(Conv2D(32,(3,3),activation='relu'))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.2))

ds_model.add(SeparableConv2D(64,(3,3),activation='relu'))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.3))

ds_model.add(SeparableConv2D(128,(3,3),activation='relu'))
ds_model.add(MaxPool2D(2,2))
ds_model.add(Dropout(0.3))

ds_model.add(Flatten())
ds_model.add(Dense(64,activation='relu'))
ds_model.add(Dropout(0.5))

ds_model.add(Dense(1,activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
ds_model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
ds_model.summary()


# In[83]:


#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


# In[15]:


history = ds_model.fit(train_generator,
                              epochs=100,
                              steps_per_epoch= len(train_generator),
                              validation_data = (valid_generator),
                              #callbacks = [early_stop]
                              #verbose=1
                              )


# In[27]:


ds_model.save(work_dir)


# In[16]:


def visualize_training(history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(history)


# In[17]:


accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

print(f'Training Accuracy: {np.max(accuracy)}')
print(f'Training Loss: {np.min(loss)}')
print(f'Validation Accuracy: {np.max(val_accuracy)}')
print(f'Validation Loss: {np.min(val_loss)}')


# As we can see DS_CNN performaned really good on training with excellent performance metrics

# ## VGG19 Model

# In[29]:


vgg_model = Sequential()
vgg_model.add(VGG19(include_top=False, pooling='avg', weights='imagenet', input_shape=(128, 128, 3), classes=2))
vgg_model.add(Flatten())
vgg_model.add(Dense(256,activation='relu'))
vgg_model.add(Dense(64,activation='relu'))
vgg_model.add(Dense(1,activation = 'sigmoid'))

vgg_model.layers[0].trainable = False

opt = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999)
vgg_model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
vgg_model.summary()


# In[31]:


#Other models that I am going to be looking at are for comparison of performance only. Thats why I am going 50 epochs

vgg_history = vgg_model.fit(train_generator,
                                      steps_per_epoch = len(train_generator),
                                      epochs=50,
                                      validation_steps = len(valid_generator),
                                      validation_data = valid_generator,
                                      verbose=1
                                     )


# In[36]:


def visualize_training(vgg_history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(vgg_history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(vgg_history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(vgg_history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(vgg_history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(vgg_history)


# In[113]:


accuracy = vgg_history.history['accuracy']
loss = vgg_history.history['loss']
val_accuracy = vgg_history.history['val_accuracy']
val_loss = vgg_history.history['val_loss']

print(f'Training Accuracy: {np.max(accuracy)}')
print(f'Training Loss: {np.min(loss)}')
print(f'Validation Accuracy: {np.max(val_accuracy)}')
print(f'Validation Loss: {np.min(val_loss)}')


# VGG19 Model is also performing really good, but compared to DS_CNN a little lower Accuracy and higher Validation Loss

# ## Inception Model

# In[37]:


inc_model = Sequential()
inc_model.add(tf.keras.applications.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_shape=(128, 128, 3), classes=2))
inc_model.add(Flatten())
inc_model.add(Dense(64,activation='relu'))
inc_model.add(Dense(1,activation = 'sigmoid'))

inc_model.layers[0].trainable = False

opt = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999)

inc_model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy'])
inc_model.summary()


# In[38]:


inc_history = inc_model.fit(train_generator,
                            steps_per_epoch = len(train_generator),
                            epochs=50,
                            validation_data=valid_generator,
                            verbose=1
                                     )


# In[43]:


inc_model_s = 'inceptionv3_malaria_predsmodel.h5'
inc_model.save(inc_model_s)


# In[116]:


def visualize_training(inc_history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(inc_history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(inc_history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(inc_history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(inc_history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(inc_history)


# In[ ]:





# In[115]:


accuracy = inc_history.history['accuracy']
loss = inc_history.history['loss']
val_accuracy = inc_history.history['val_accuracy']
val_loss = inc_history.history['val_loss']

print(f'Training Accuracy: {np.max(accuracy)}')
print(f'Training Loss: {np.min(loss)}')
print(f'Validation Accuracy: {np.max(val_accuracy)}')
print(f'Validation Loss: {np.min(val_loss)}')


# Looks like Inception model is not performing really well. We can clearly see the validation loss skyrocketting in the end of training cycle. That is obviously situation of Overfitting

# ## Evaluating Models on Test Dataset

# In[85]:


#eval_datagen = ImageDataGenerator(rescale=1./255)
eval_generator = test_datagen.flow_from_directory(directory = test_dir,
                                                   target_size=(128, 128),
                                                   class_mode=None,
                                                   shuffle = False,
                                                   batch_size=32
                                                 )


# #### Depth-Wise Separable CNN (DS-CNN) evaluation

# In[100]:


eval_generator.reset()    
pred_ds_cnn = ds_model.predict(eval_generator, 1000, verbose=1)
print("Predictions finished")


# In[101]:


import matplotlib.image as mpimg

for index, probability in enumerate(pred_ds_cnn):
    image_path = test_dir + "/" +eval_generator.filenames[index]
    img = mpimg.imread(image_path)

    plt.imshow(img)
    print(eval_generator.filenames[index])
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% B")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% A")
    plt.show()
    
 
    


# #### VGG19 Model evaluation

# In[105]:


eval_generator.reset()    
pred_vgg = vgg_model.predict(eval_generator, 50, verbose=1)
print("Predictions finished")


# In[106]:


import matplotlib.image as mpimg

for index, probability in enumerate(pred_vgg):
    image_path = test_dir + "/" +eval_generator.filenames[index]
    img = mpimg.imread(image_path)

    plt.imshow(img)
    print(eval_generator.filenames[index])
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% B")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% A")
    plt.show()
    
 
    


# #### Inception Model evaluation

# In[102]:


eval_generator.reset()    
pred_inception = inc_model.predict(eval_generator, 50, verbose=1)
print("Predictions finished")


# In[103]:


import matplotlib.image as mpimg

for index, probability in enumerate(pred_inception):
    image_path = test_dir + "/" +eval_generator.filenames[index]
    img = mpimg.imread(image_path)

    plt.imshow(img)
    print(eval_generator.filenames[index])
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% B")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% A")
    plt.show()
    
 
    


# ## Trying one more CNN model but using OpenCV for Image Preprocessing

# In[117]:


IMG_SIZE = 120
CATEGORIES = ['Parasitized', 'Uninfected']
dataset = []

def generate_data():
    for category in CATEGORIES:
        path = f'dataset/cell_images/{category}'
        class_id = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
                image_array = cv2.resize(image_array, (IMG_SIZE , IMG_SIZE))
                dataset.append([image_array, class_id])
            except Exception as e:
                print(e)
    random.shuffle(dataset)
                
generate_data()


# In[118]:


print(len(dataset))


# In[119]:


data = []
labels = []
for features, label in dataset:
    data.append(features)
    labels.append(label)

data = np.array(data)
data.reshape(-1, 120, 120, 3)


# In[120]:


train_data, data, train_labels, labels = train_test_split(data, 
                                                          labels,
                                                          test_size=0.15)

test_data, validation_data, test_labels, validation_labels = train_test_split(data, 
                                                                    labels,
                                                                    test_size=0.7)


# In[121]:


plt.figure(figsize=(10, 10))
i = 0
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_data[i])
    if(test_labels[i] == 0):
        plt.xlabel('Infected')
    else:
        plt.xlabel('Uninfected')
    i += 1
plt.show()


# In[122]:


#Image Augmentation
datagen_train = ImageDataGenerator(rescale=1./255,
                            rotation_range=45,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

datagen_test = ImageDataGenerator(rescale=1./255)
datagen_validation = ImageDataGenerator(rescale=1./255)


# In[123]:


datagen_train.fit(train_data)
datagen_test.fit(test_data)
datagen_test.fit(validation_data)


# In[126]:


cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPool2D((2, 2)),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPool2D((2, 2)),
    
    Conv2D(128, (3, 3), activation="relu"),
    MaxPool2D((2, 2)),
    
    Conv2D(256, (3, 3), activation="relu"),
    MaxPool2D((2, 2)),
    
    Flatten(),
    Dense(256, activation="relu"),
    Dense(2, activation='softmax')
])
cnn_model.summary()


# In[128]:


cnn_model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[129]:


history = cnn_model.fit_generator(datagen_train.flow(train_data, train_labels, batch_size=32),
                   steps_per_epoch=len(train_data) / 32,
                   epochs=50,validation_data=datagen_validation.flow
                              (validation_data,validation_labels, batch_size=32),                    
                   )


# In[131]:


def visualize_training(history, lw = 3):
    plt.figure(figsize=(10,6))
    plt.plot(inc_history.history['accuracy'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(inc_history.history['val_accuracy'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize = 'x-large')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(inc_history.history['loss'], label = 'training', marker = '*', linewidth = lw)
    plt.plot(inc_history.history['val_loss'], label = 'validation', marker = 'o', linewidth = lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize = 'x-large')
    plt.show()
visualize_training(inc_history)


# In[134]:


accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

print(f'Training Accuracy: {np.max(accuracy)}')
print(f'Training Loss: {np.min(loss)}')
print(f'Validation Accuracy: {np.max(val_accuracy)}')
print(f'Validation Loss: {np.min(val_loss)}')


# In[143]:


random.shuffle(test_data)
predictions = cnn_model.predict(test_data)


# In[135]:


class_names = ['Infected', 'Uninfected']
def plot_images(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i],images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img)
    
    predicted_label = np.argmax(predictions_array)
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]))


# In[141]:


num_rows = 100
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_images(i, predictions, test_labels, test_data)

