import os
import cv2
import keras.layers as layers
from keras.utils import np_utils
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


def mapper(label):
    """
    Retrieves the numerical representation of label stored LABEL_MAP.
    Parameters:
      * label: label of image (E.g. rock/paper/etc)
    Return: Numerical represntation of label.
    """
    return LABEL_MAP[label]


def prepare_data(dataset):
    """
    Prepares data into suitabe formats to be used by the model.
    Seperates images and labels from dataset, conducts numerical conversion
    of labels and subsequent one-hot encoding.
    Parameters:
      * dataset: dataset containing images and labels lists.
    Return: seperated and processed images and one-hot encoded labels.
    """
    # dataset = [[img, 'rock'], [img, 'paper], [img, 'paper'], ...]
    # zip(*list) - passing elements of nested list as seperate args.
    # In assignment unpacking can lead to another list - E.g. a, b = [1, 2, 3] -> a = 1 and b = [2, 3]
    # however when unpacking within a function call the elements are unpacked and passed as seperate arguments to function
    # rather than collecting them into a new list.  zip([img, 'rock'], [img, 'paper'], ...) 
    images, labels = zip(*dataset)  # [img, img, ...], ['rock', 'paper', ...]
    
    # Mapping labels to numbers.
    labels = list(map(mapper, labels))

    # One-hot encoding the labels.
    labels = np_utils.to_categorical(labels)
    # one hot encode the labels  # part of preparing the data.
    # Converts list of numerical labels into one-hot encoded vectors (binary vectors).
    # Each category is represented by a vector where all elements are 0 except for the index
    # corresp. to category - E.g. rock = [1, 0, 0, 0], paper [0, 1, 0, 0], scissors [0, 0, 1, 0], none [0, 0, 0, 1]
    # one-hot encoding is needed for categorical variables when training ML models esp. neural networks.
    # as they require numerical input (on-hot encoding represents categorical variables numerically) in a format
    # suitable for training.
    # none is also one-hot encoded as its incl. in class_map and numerical labels.

    return images, labels

    


