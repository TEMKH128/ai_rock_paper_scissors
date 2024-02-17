import os
import cv2
import numpy
import keras.layers as layers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.applications.densenet import DenseNet121


LABEL_MAP = {
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
    DenseNet121 as the base model (for feature extraction). Layers added
    to model and model (used for image classification) is returned.
    """
    # DenseNet121 model used as base layer (include_top = false, don't incl.
    # top classification layer), add our our own layers.
    # weights="..." (pre-trained weights [parameters model learns during process] trained on ImageNet dataset will be loaded into
    # model.
    # classes=4: model trained to classify images into 4 classes.
    # input_shape=... specifies shape of input images (224 x 224 pixels /w 3 colour channels)
    # base_model.trainable=True allows fine-tuning of base model's weight during training.
    # Sequential - keras model that creates a linear stack of layers (structure of neural network), layers added to sequential
    # model are arranged sequentially (one after another) - output of 1 layer serves as input for next layer (flow of data from input layer to hidden layers to output layer)
    # model.add(layers.MaxPool2D()) reduces spacial dimension of feature maps (shapes, texture, etc) helps reduce computational complexity and overfitting.
    # layers.Flatten(): Flattens multi-dimensioanl output from previous layer into 1-dimensional vector, which is required before passing to a fully connected layer. [E.g. converting from grid to simple list]
    # layers.Dense(num_classifications, activation='softmax') adds a Dense (fully connected) layer to the model with a softmax activation function.
    # no. of units in layer = num._classification, softmax activation function to the output of the previous layer. Softmax converts the raw scores into probability distributions over the different classes.

    base_model = DenseNet121(include_top=False, weights='imagenet', classes=4, input_shape=(224, 224, 3))
    base_model.trainable = True  # Set to False if you don't want to fine-tune

    model = Sequential()
    model.add(base_model)
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classifications, activation='softmax'))
    
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
            img = cv2.imread(os.path.join(label_path, item))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # 224 x 224 pixels, Squeezenet.
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
    labels = to_categorical(labels)
    # one hot encode the labels  # part of preparing the data.
    # Converts list of numerical labels into one-hot encoded vectors (binary vectors).
    # Each category is represented by a vector where all elements are 0 except for the index
    # corresp. to category - E.g. rock = [1, 0, 0, 0], paper [0, 1, 0, 0], scissors [0, 0, 1, 0], none [0, 0, 0, 1]
    # one-hot encoding is needed for categorical variables when training ML models esp. neural networks.
    # as they require numerical input (on-hot encoding represents categorical variables numerically) in a format
    # suitable for training.
    # none is also one-hot encoded as its incl. in class_map and numerical labels.

    return images, labels

    
def configure_model(model):
    """
    Configures model for training, specifying optimisation algorithm (Adam)
    with its learning rate, loss function used to evalaute model performance,
    and metrics used to evaluate model performance.
    Parameters:
      * model: model to be configured.
    Return: configured model.
    """

    # retrieve_model called from outsided and passed into configure model.
    # compile - configures learning process (model) for training.
    # optomizer - Specifies the optimisation algorithm to be dused during training (Adam(lr=0.0001)).
    # Adam is an optimisation algorithm (used to find the best possible solution to a given problem),
    # loss -specifies loss function used to eval. model's performanace during training.
    # categorical-crosentropy is common for multi-class classificatin with one-hot encoded labels.
    # metrics - specifies metrics to be used for eval. model's performance during training and testing.
    # accuracy metric will be calc. and displayed during training.
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, images, labels):
    """
    Training the model, using the images (data) and their corresponding
    one-hot encoded labels over specified epochs (iterations)
    Parameters:
      * model: model to be trained.
      * images: data images list used in training.
      * labels: one-hot encoded labels list that corresponds to images list.
    Return: The trained model.
    """
    # why train over 8 epochs/iterations. The less the loss and higher the accuracy the better the model (goal of each epoch)
    # training for too few epochs may result in underfitting, where the model fails to capture the underlying patterns in the data.
    # Conversely, training for too many epochs may lead to overfitting, where the model performs 
    # well on the training data but fails to generalize to unseen data. 10 epochs choice is based on
    # empirical experimentation or prior knowledge of the dataset's complexity.
    # lists correspond with one another - index 0 of labels belong to index 0 of data.
    # model.fit() takes input data and its corresp. labels and does the training over a specified no. of epochs.
    # Batch size: no. of training examples (samples of data) used in one epoch/iteration, 
    # using batches allows model to process data more efficiently, smaller batch sizes consume less memory.
    # Epoch: 1 iteration or complete pass through entire training dataset (all batches per epoch), batches are randomised with each epoch. (To improve ability to generalise unseen data and avoid overfitting)
    model.fit(numpy.array(images), numpy.array(labels), batch_size=8, epochs=5)
    return model


def save_trained_model(model):
    """
    Save the trained model for later use following Hierarchical Data
    Format (HDF) which stores large amounts of numerical data.
    Parameters:
      * model: Trained model to be saved.
    """
    model.save("trained_models/rock-paper-scissors.keras")
    print("Model saved to trained_model directory within project.")


def execute_model_training():
    dataset = load_collected_images("image_data")
    images, labels = prepare_data(dataset)
    
    model = retrieve_model(len(LABEL_MAP))
    model = configure_model(model)
    model = train_model(model, images, labels)
    save_trained_model(model)


if __name__ == "__main__":
    execute_model_training()