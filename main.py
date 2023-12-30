
import os, sys

import numpy as np
from infer_ocr import infer_ocr
from capture_text import findSpeechBubbles


class action_manga():
    def __inif__(self, folder_path):
        self.ocr_engine = infer_ocr()
        self.folder_path = folder_path

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
            text_res = self.ocr_engine.text_generator(img_path)
            

        return 
    
if __name__ == "__main__":
    test = action_manga()
    test.main()