
import os, sys

import numpy as np
from infer_ocr import infer_ocr
from capture_text import findSpeechBubbles


class action_manga():
    def __init__(self, folder_path):
        self.ocr_engine = infer_ocr()
        self.folder_path = folder_path
        self.crop_textPath = folder_path + "/cropped"
        if not os.path.isdir(self.crop_textPath):
            os.makedirs(self.crop_textPath)

    def load_data_path(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        image_paths = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                _, extension = os.path.splitext(file)
                if extension.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))

        return image_paths 

    def main(self):
        image_path = self.load_data_path()
        for img_path in image_path:
            croppedImageData = findSpeechBubbles(img_path, "complex", self.crop_textPath)

        croppedImageList = os.listdir(self.crop_textPath)
        for crop_img in croppedImageList:   
            text_res = self.ocr_engine.text_generator(self.crop_textPath + "/" + crop_img)

        return 
    
if __name__ == "__main__":
    folder_path = "data/testing_data"
    test = action_manga(folder_path)
    test.main()