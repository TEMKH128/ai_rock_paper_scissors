import cv2
import numpy
import unittest
from keras.models import load_model


class TestTrainedModel(unittest.TestCase):
    number_label_map = {
        0: "rock",
        1: "paper",
        2: "scissors",
        3: "none"
    }

    model = load_model("trained_models/rock-paper-scissors-test-model.keras")
    def test_rock_classification(self):
        # Prepare image.
        img = cv2.imread("test_image_data/rock/rock_1.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # Predict move.
        # makes predictions using a ML model based on input ('img').
        # converts image into a numpy array and wraps it in another array as predict() expects
        # array-like input.
        predictions = self.model.predict(numpy.array([img]))
        # np.argmax() returns the index of the max. value in an array. [one-hot encoded label - [1 0 0 0] max value index is 0 - rock]
        move_code = numpy.argmax(predictions[0])  # corresponds with numpy array index from predict()
        label = self.number_label_map[move_code]

        self.assertEqual("rock", label)

    def test_paper_classification(self):
        # Prepare image.
        img = cv2.imread("test_image_data/paper/paper_2.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # Predict move.
        predictions = self.model.predict(numpy.array([img]))
        move_code = numpy.argmax(predictions[0])
        label = self.number_label_map[move_code]

        self.assertEqual("paper", label)
        
    def test_scissors_classifcation(self):
        # Prepare image.
        img = cv2.imread("test_image_data/scissors/scissors_8.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # Predict move.
        predictions = self.model.predict(numpy.array([img]))
        move_code = numpy.argmax(predictions[0])
        label = self.number_label_map[move_code]

        self.assertEqual("scissors", label)
        
    def test_none_classification(self):
        # Prepare image.
        img = cv2.imread("test_image_data/none/none_9.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # Predict move.
        predictions = self.model.predict(numpy.array([img]))
        move_code = numpy.argmax(predictions[0])
        label = self.number_label_map[move_code]

        self.assertEqual("none", label)
        

if __name__ == "__main__":
    unittest.main()