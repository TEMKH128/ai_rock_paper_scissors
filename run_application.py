import cv2
import numpy
from keras.models import load_model

NUMBER_LABEL_MAP = {  # Used to classiy predicition.
        0: "rock",
        1: "paper",
        2: "scissors",
        3: "none"
    }


def determine_winner(player_move, comp_move):
    """
    Determines the winner of the rock-paper-scissors game between
    player and computer.
    Parameters:
      * user_move: move executed by user (R/P/S).
      * player_move: move executed by computer (R/P/S).
    Return: outcome of move - tie/computer/player.
    """
    player_move = player_move.lower()
    comp_move = comp_move.lower()

    if (player_move == comp_move):
        return "tie"

    # Computer wins.
    elif ((player_move == "rock" and comp_move == "scissors") or
        (player_move == "paper" and comp_move == "rock") or
        (player_move == "scissors" and comp_move == "paper")):

        return "player"
    
    else:  ## Computer wins.
        return "computer"

    
def load_trained_model():
    """
    Load trained rock-paper-scissors model that is aved in trained_model
    package.
    Return: Return the model loaded.
    """
    try:
        model = load_model("trained_models/rock-paper-scissors-model.h5")

    except OSError:
        print("model not found within trained_models package")
        return None

def create_playing_areas(frame):
    """
    Creates playing areas to be used by player (capture image) and
    computer (Display its move).
    Parameters:
      * frame: frame rectangles will be drawn on.
    """
    # User's play area.
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)  # BGR - white.

    # Computer's play area.
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)  # BGR - white.


def extract_user_image(frame):
    """
    Extracts region of interest (Use'rs portion of frame) from frame.
    Convert its colour space form BGR to RGB and resize image to 224 x 224,
    which is ideal when working with DenseNet model.
    Parameters:
      * frame: frame where region of interest will be extracted from.
    Return: image representing region of interest.
    """
    # Extract region of interest (roi) from frame (rectangle) using
    # numpy array slicing - rows and column (100th to 499th).
    region_of_interest = frame[100:500, 100:500]

    img = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # 224 x 224 pixels, DenseNet.

    return img


def predict_player_move(model, img):
    """
    Uses model and image captured (represents player's move) to predict
    the players move (rock/paper/scissors).
    Parameters:
      * model: trained model that will classify image.
      * img: image representing players move to be classified.
    Return: player's predicted/classified move.
    """
    # Predict move.
    # makes predictions using a ML model based on input ('img').
    # converts image into a numpy array and wraps it in another array as predict() expects
    # array-like input.
    predictions = model.predict(numpy.array([img]))
    # np.argmax() returns the index of the max. value in an array. [one-hot encoded label - [1 0 0 0] max value index is 0 - rock]
    move_code = numpy.argmax(predictions[0])  # corresponds with numpy array index from predict()
    
    return NUMBER_LABEL_MAP[move_code]


def execute_program():
    model = load_trained_model()
    if (model == None): return

    # Initialise video capture object, opens default camera (0), which will be
    # used to capture video frames from camera.
    capture = cv2.VideoCapture(0)

    while (True):
        # Reads a frame (particular instance of video in single point in time,
        # treated like images) from video capture object.  
        # tuple returned (frame_retrieved_boolean, image/frame [numpy array])
        retrieved, frame = capture.read()

        if (not retrieved): continue

        create_playing_areas(frame)
        img = extract_user_image(frame)
        player_move = predict_player_move(model, img)
