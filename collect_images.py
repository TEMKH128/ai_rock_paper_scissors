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

def collect_images():
    args = retrieve_arguments()

    if (len(args) == 2):
        dest_path = create_image_directories("iamge_data", args[0])


if __name__ == "__main__":
    collect_images()