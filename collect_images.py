description = """
Gathers data images (with specified label) to be used to train model.
Usage: python3 collect_images.py <label_name> <sample_size>  E.g. python3 rock 200.
Order <label_name> <sample_size> is important!

Only portion of image within the box displayed will be collected.

Start/Pause Collection Process: Enter/Hit "spacebar".
Stop Collection Process: Enter/Hit "q".
"""

import os
import cv2
import sys


def retrieve_arguments():
    """
    Retrieves required command line arguments <label_name> <sample_size>
    Return: List containing retrieved command-line arguments, empty list
    if none are retrieved.
    """
    args = []
    try:
        args.append(sys.argv[1])
        args.append(int(sys.argv[2]))

    except:
        print("Error: Needed arguments are missing.\n" + description)

    return args


def create_image_directories(img_dir, label):
    """
    Create image directory and label directories (within image dir) which
    will be used to store collected images.
    Parameters: 
        * img_dir: name of the image directory to be created.
        * label: name of specific label (E.g. rock) directory to be creatd. 
    Return: Destination path where images will be stored.
    """
    label_path = os.path.join(img_dir, label)

    try:
        os.mkdir(img_dir)

    except FileExistsError:
        pass

    try:
        os.mkdir(label_path)

    except FileExistsError:
        print(f"{label_path} already exists, collected images will be " +
            "added to existing folder.")
        
    return label_path


def extract_save_image(frame, dest_path, label, image_count):
    """
    Extracts region of interest from frame and attempts to
    save it as an image within the destination path.
    Parameters:
      * frame: Used to extract region of interest (roi).
      * dest_path: Where images will be saved.
      * label: Current image type/label being collected.
      * image_count: Number of images that have been collected thus far.
    Return: Number of images that have been collected thus far.
    """
    # Extract region of interest (roi) from frame (rectangle) using
    # numpy array slicing - rows and column (100th to 499th).
    region_of_interest = frame[100:500, 100:500]

    image_path = os.path.join(dest_path, f"{label}_{image_count +1}" +
        ".jpg")  # file path must include image format.
    
    # Saiving roi as an image - cv2.imwrite(filename, image), return true
    # if successful.
    if (cv2.imwrite(image_path, region_of_interest)): image_count +=1

    return image_count


def display_frame(frame, image_count, num_images, label):
    """
    Draws text string on frame/image and displays frame in a window.
    Parameters:
      * frame: frame to be displayed and where text will be written on.
      * image_count: Number of images that have been collected thus far.
      * num_images Total number of images to be collected.
      * label: Current image type/label being collected.
    """
    #cv2.putText(image, text, bottom_left_pos, font, font_scale 
    # [relative to base font size of font type], text_thickness, 
    # opt_line_type). cv2.LINE_AA = antialised (process for smoothing
    # lines = higher quality text)
    cv2.putText(frame, f"Images Collected: {image_count}/{num_images}",
        (5, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255),
        2, cv2.LINE_AA)
    
    # Displays frame with window text.
    cv2.imshow(f"Collecting {label} Images:", frame)


def check_user_input(capture_image, quit_process):
    """
    Checks for user key presses and responds appropriately to them.
    Parameters:
      * capture_image: bool reflecting whether image should be captured / not.
      * quit_proccess: bool reflecting whether image capture process should
        be exited.
    Return: capture_image and quit_process.
    """
    # Waits for key press for duration E.g. 10 milliseconds then moves on,
    # shorter waits better if need to respond quickly to user and for 
    # higher frame rates but consumes more CPU resources.
    # If wait time 0 = waits indefinitely until key pressed, > 0 for time
    # specified, < 0 indefinitely w/o blocking program.
    # Returns ASCII (numerical) value of key press, If none within time
    # period than returns -1.

    key_press = cv2.waitKey(10)
    if (key_press == ord(" ")):  # " " represents spacebar.
        capture_image = not capture_image

    elif (key_press == ord("q")): quit_process = not quit_process
    
    return capture_image, quit_process


def release_resources(capture):
    """
    Releases resources used during image capturing process.
    Parameters:
      * capture: video capture object used to capture video frames.
    """
    # Release video capture resourcs, opencv windows (cv2.imshow())
    capture.release()
    cv2.destroyAllWindows()


def capture_images(dest_path, label, num_images):
    """
    Captures images and stores them in the provided file path.
    Parameters:
      * dest_path: path where images will be stored.
      * label: Type of images being captured (rock/paper/scissors/etc)
      * num_images: number of images that should be captured.
    """
    # Initialise video capture object, opens default camera (0), which will be
    # used to capture video frames from camera.
    capture = cv2.VideoCapture(0)
    capture_image = quit_process = False
    image_count = 0

    while (image_count < num_images and (not quit_process)):
        # Reads a frame (particular instance of video in single point in time,
        # treated like images) from video capture object.  
        # tuple returned (frame_retrieved_boolean, image/frame [numpy array])
        retrieved, frame = capture.read()

        if (not retrieved): continue

        # Draws rectangle on frame from (100, 100) to (500, 500) that is white
        # and has a line thickness of 2 pixels.
        cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

        if (capture_image):
            image_count = extract_save_image(frame, dest_path, label,
                image_count)
            
        display_frame(frame, image_count, num_images, label)
        capture_image, quit_process = check_user_input(capture_image, quit_process)

    print(f"{num_images} Image(s) Saved to {dest_path}.")
    release_resources(capture)


def collect_images():
    args = retrieve_arguments()

    if (len(args) == 2):
        dest_path = create_image_directories("image_data", args[0])
        capture_images(dest_path, args[0], args[1])


if __name__ == "__main__":
    collect_images()