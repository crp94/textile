import cv2
import kornia as K

def read_image(image_path):
    return cv2.imread(image_path, -1)[:,:,:3][:,:,::-1] / 255

def process_image(image):
    return K.image_to_tensor(image).unsqueeze(0)

def read_and_process_image(image_path):
    return process_image(read_image(image_path))