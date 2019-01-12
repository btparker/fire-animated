import platform

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

import skimage

import sklearn
from sklearn.preprocessing import scale

import numpy as np

import tensorflow
from tensorflow import keras
from keras.layers import (BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation)
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Input, InputLayer
from keras.models import Model
import keras.backend as K
from keras.models import load_model
import coremltools


from livelossplot import PlotLossesKeras
print("Python version:", platform.python_version())
print("Matplotlib version:", matplotlib.__version__)
print("SKImage version:", skimage.__version__)
print("SKLearn version:", sklearn.__version__)
print("Numpy version:", np.__version__)
print("Tensorflow version:", tensorflow.__version__)
print("Keras version:", keras.__version__)

### IMAGE HELPERS ###
TARGET_IMG_SIZE = (738, 960)
TARGET_IMG_RGB_SIZE = (*TARGET_IMG_SIZE, 3)

## TRAINING HELPERS
# Which things to 'find' in an image, must have
# the same name in training directory
DETECTION_CLASSES = ["Fire"]

# Adding background to better train data
TRAIN_CLASSES = DETECTION_CLASSES + ["Background"]

LABEL2CLASS = dict(enumerate(TRAIN_CLASSES))
CLASS2LABEL = {cl: label for label, cl in LABEL2CLASS.items()}

CLASS_WEIGHTS = {label: 1 if cl != 'Background' else 1 for label, cl in LABEL2CLASS.items()}

def get_data_and_labels(directory, shuffle=True):
    gen = image.ImageDataGenerator()
    batches = gen.flow_from_directory(directory, target_size=TARGET_IMG_SIZE, batch_size=1, shuffle=shuffle)

    imgs = []
    labels = []
    for i in range(batches.n):
        img, label = batches.next()
        imgs.append(img)
        labels.append(label)

    data = np.concatenate(imgs)
    labels = np.concatenate(labels)
    
    return data, labels

def display_image_and_heatmap(images, heatmaps, predictions, number):
    predicted_label = np.argmax(predictions[number])
    predicted_class = LABEL2CLASS[predicted_label]
    print("Predicted class:", predicted_class)
    for label, prediction in enumerate(predictions[number]):
        print("{} confidence: {}%".format(predicted_class, round(prediction * 100, 2)))

    fig = plt.figure(figsize=(18, 18))
    img = images[number].astype(np.uint8)

    for cl in DETECTION_CLASSES:
        label = CLASS2LABEL[cl]
        fig.add_subplot(2, 1, label + 1, title="Heatmap for " + cl + " recognition (predicted "+ predicted_class + ")")
        heat_map_low_res = np.moveaxis(heatmaps[number], 2, 0)[label]
        heat_map = skimage.transform.resize(heat_map_low_res, TARGET_IMG_SIZE, mode="reflect", anti_aliasing=True)
        plt.imshow(img)
        plt.imshow(heat_map, cmap="seismic", alpha=0.5, vmax=heatmaps[number].max())
        plt.colorbar()
        plt.savefig('image_{}.png'.format(idx))



def create_model_and_fit(
    train_data, train_labels, val_data, val_labels, nb_classes,
                        nb_epochs,record_input_global_average_pooling=False):

    
    vgg16_bottom = VGG16(include_top=False, input_shape=TARGET_IMG_RGB_SIZE, weights='imagenet')
    model_bottom = Sequential(VGG16(include_top=False, input_shape=TARGET_IMG_RGB_SIZE, weights='imagenet').layers[:-1])
    
    print("Predict output of non trainable layers: Training set")
    post_model_bottom_train_features = model_bottom.predict(train_data, batch_size=1, verbose=1)
    
    print("Predict output of non trainable layers: Validation set")
    post_model_bottom_val_features = model_bottom.predict(val_data, batch_size=1, verbose=1)
    
    model_top = Sequential([
                Conv2D(nb_classes, (3, 3), activation='relu', padding='same', name="conv_last"),
                GlobalAveragePooling2D(name="global_average_pooling"),
                Activation("softmax", name="softmax")
            ])
    
    model_top.compile(loss=categorical_crossentropy, optimizer=Adam(1e-3), metrics=[categorical_accuracy])

    print("Fit the last convolutional layer")
    model_top.fit(post_model_bottom_train_features, train_labels,
                      validation_data=(post_model_bottom_val_features, val_labels),
                      epochs=nb_epochs, verbose=0, batch_size=1,
                      class_weight = CLASS_WEIGHTS)        
    
    inp = x = Input(shape=TARGET_IMG_RGB_SIZE)

    for layer in model_bottom.layers + model_top.layers:
        if layer.name == 'global_average_pooling':
            conv_last = x
        x = layer(x)
    
    model = Model([inp], [x, conv_last])    
    
    return model

training_directory="data/dataset/train"
validation_directory="data/dataset/validation"

train_data, train_labels = get_data_and_labels(training_directory)
val_data, val_labels = get_data_and_labels(validation_directory)

# model = create_model_and_fit(
#   train_data=train_data,
#   train_labels=train_labels,
#   val_data=val_data,
#   val_labels=val_labels,
#   nb_classes=2,
#   nb_epochs=2000,
# )

# model.summary()

# # Save entire model to a HDF5 file
# model.save('model.h5')
# del model
model = keras.models.load_model('model.h5')
coreml_model = coremltools.converters.keras.convert('model.h5')
coreml_model.save('model.mlmodel')

# val_data, val_labels = get_data_and_labels(validation_directory, False)

val_predictions, val_last_conv = model.predict(val_data, verbose=1, batch_size=3)

for idx in range(0, len(val_predictions)):
    display_image_and_heatmap(val_data, val_last_conv, val_predictions, idx)