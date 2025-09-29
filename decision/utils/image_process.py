import base64
import cv2
import json
import os

def image_to_base64_256(img):
    """
    Convert an image to Base64 encoding
    """
    img_small = cv2.resize(img, (256, 256))  # Resize to 256x256 for example
    _, buffer = cv2.imencode('.png', img_small)
    return base64.b64encode(buffer).decode()

def image_to_base64_128(img):
    """
    Convert an image to Base64 encoding
    """
    img_small = cv2.resize(img, (128, 128))  # Resize to 128x128 for example
    _, buffer = cv2.imencode('.png', img_small)
    return base64.b64encode(buffer).decode()


def image_path_to_base64(image_path):
    """
    Convert an image file path to Base64 encoding
    :param image_path: Path to the image file
    :return: Base64 encoded string
    """
    # Check if path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image, check if the path is correct or if the file is corrupted: {image_path}")
    
    # Call image_to_base64 function for encoding
    return image_to_base64_128(img)

def process_camera_output(state):
    """
    Process multiple camera outputs and return a dictionary of Base64 encoded images
    """
    camera_images = {}
    camera_keys = ["LeftCamera", "RightCamera", "DownCamera", "UpCamera", "FrontCamera", "BackCamera"]

    # camera_keys = ["FrontCamera"]
    for camera in camera_keys:
        if camera in state:
            pixels = state[camera]
            cv2.imshow(f"{camera} Output", pixels[:, :, 0:3])
            cv2.waitKey(1)
            camera_images[camera] = image_to_base64_128(pixels)

    return camera_images if camera_images else None

def save_base64_images(b64_images, output_file):
    """
    Save Base64 encoded images to a file
    """
    with open(output_file, "w") as f:
        json.dump(b64_images, f)