import cv2
import numpy
import random
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
        # "trained_models/rock-paper-scissors-model.keras"
        return load_model("trained_models/rock-paper-scissors-test-model.keras")

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


def determine_player_comp_winner(previous_move, player_move, computer_move, winner):
    """
    Determines winner provided that the previous move made by player isn't
    their current move and their move is classified as Rock/Paper/Scissors.
    Parameters:
      * previous_move: player's previous move.
      * player_move: player's current move.
      * computer_move: stores computer's most recent move.
      * winner: winner if outcome can be determined otherwise error message.
    Return: computer's most recent move and winner 
    (player/computer/error-message).
    """
    if (previous_move != player_move and player_move != "none"):
        computer_move = random.choice(['rock', 'paper', 'scissors'])
        winner = determine_winner(player_move, computer_move)
        
    elif (previous_move != player_move and player_move == "none"):
        computer_move = None
        winner = "Undetermined...No Moves Made."

    return computer_move, winner


def display_frame(frame, player_move, computer_move, winner):
    """
    Displays frame where user make's moves, display's computer's moves using
    icons (Rock/Paper/Scissors) and displays text reflecting game state
    (i.e. who won, etc).
    Parameters:
      * frame: frame to displayed.
      * player_move: move made by player (Rock/Paper/Scissors/None).
      * computer_move: move made by computer (Rock/Paper/Scissors/None).
      * winner: outcome of game (Player/Computer/None)
    """
    cv2.putText(frame, f"Player Move: {player_move}", (50, 50), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"Computer Move: {computer_move}", (750, 50), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"Winner: {winner}", (400, 600), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, (87, 139, 46), 2, cv2.LINE_AA)
    
    if (computer_move != None):
        print(f"move_icons/{computer_move.lower()}.png")
        move_icon = cv2.imread(f"move_icons/{computer_move.lower()}.png")
        print(move_icon.shape)
        
        # Base off Computer's playing area (cv2.rectangle)
        move_icon = cv2.resize(move_icon, (400, 400))
        print(move_icon.shape)
        print("Shape of frame before slicing:", frame.shape)
        frame[100:500, 800: 1200] = move_icon

    cv2.imshow("AI Rock Paper Scissors", frame)


def execute_program():
    print("Attempting to Load Model ...")
    model = load_trained_model()
    
    if (model == None): return

    # Initialise video capture object, opens default camera (0), which will be
    # used to capture video frames from camera.
    # Set resolution to 1280x720 (More space)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous_move = computer_move = winner = None
    while (True):
        # Reads a frame (particular instance of video in single point in time,
        # treated like images) from video capture object.  
        # tuple returned (frame_retrieved_boolean, image/frame [numpy array])
        retrieved, frame = capture.read()

        if (not retrieved): continue

        create_playing_areas(frame)
        img = extract_user_image(frame)
        player_move = predict_player_move(model, img)
        print("184: " + player_move)

        # Determine winner (player/computer).
        computer_move, winner = determine_player_comp_winner(previous_move,
            player_move, computer_move, winner)

        previous_move = player_move

        display_frame(frame, player_move, computer_move, winner)


if __name__ == "__main__":
    execute_program()