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
    # DenseNet121 model used as base layer for model.
    base_model = DenseNet121(include_top=False, weights="imagenet", classes=4,
      input_shape=(224, 224, 3))
    
    # fine-tune base model's weigth.
    base_model.trainable = True

    # Linear stack of layers and subsequent layers.
    model = Sequential()
    model.add(base_model)
    model.add(layers.MaxPool2D())  # reduce spatial dimensions of feature map.
    model.add(layers.Flatten())  # Flatten multi-dimensional output.

    # Add Dense (fully connected) layer to model.
    model.add(layers.Dense(num_classifications, activation="softmax"))
    
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

            # Read image from specified image path.
            img = cv2.imread(os.path.join(label_path, item))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # 224 x 224 pixels, Densenet.
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
    # Unpack dataset to images and labels list. 
    images, labels = zip(*dataset)
    
    # Mapping labels to numbers.
    labels = list(map(mapper, labels))

    # One-hot encoding the labels.
    labels = to_categorical(labels)

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
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
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
    # Train model with images and labels over specified epochs/iteration
    # in specified number of batches.
    model.fit(numpy.array(images), numpy.array(labels), batch_size=8,
      epochs=5)
    
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