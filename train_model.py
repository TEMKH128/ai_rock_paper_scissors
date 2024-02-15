import os
import cv2
import keras.layers as layers
from keras.models import Sequential
from keras_squeezenet import SqueezeNet


LABEL_MAP = {  # used also for num_classifications.
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}


def retrieve_model(num_classifications):
    """
    Parameters:
      * num_classifications: number of classifications possible.
    Defines Convolution Neural Network (CNN) model architecture using
    SqueezeNet as the base model (for feature extraction). Layers added
    to model and model (used for image classification) is returned.
    """
    linear_stack_layers = [
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        layers.Dropout(0.5),
        layers.Convolution2D(num_classifications, (1, 1), padding='valid'),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        layers.Activation('softmax')
    ]

    model = Sequential(linear_stack_layers)
    return model


def load_collected_images(img_dir):
    """
    Retrieves images from image directory and resizes those images and
    chang their colour space to RGB and stores each image in a list.
    Parameters:
      * img_dir: directory used to extract images it contains.
    Return: dataset.
    """
    dataset = []
    for directory in os.listdir(img_dir):
        label_path = os.path.join(img_dir, directory)

        if (not os.path.isdir(label_path)): continue

        for item in os.listdir(label_path):
            if (item.startswith(".")): continue  # excl. hidden files.

            # read image from specified image path.
            img = cv2.imread(os.path.join(label_path), item)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (227, 227))  # 227 x 227 pixels, Squeezenet.
            dataset.append([img, directory])

    return dataset

    


