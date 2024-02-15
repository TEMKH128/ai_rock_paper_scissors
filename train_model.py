import keras.layers as layers
from keras.models import Sequential
from keras_squeezenet import SqueezeNet


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


