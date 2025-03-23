
import os, sys

import numpy as np
from infer_ocr import infer_ocr
from capture_text import findSpeechBubbles
from manga_seg import seg_test
from utils import convert_to_jpg
import cv2
from translate import M2M_translate

class action_manga():
    def __init__(self, folder_path):
        self.ocr_engine = infer_ocr()
        self.folder_path = folder_path
        self.crop_textPath = folder_path + "/cropped/"
        if not os.path.isdir(self.crop_textPath):
            os.makedirs(self.crop_textPath)
        self.save_translate_txt_path = folder_path + "/translated/"
        if not os.path.isdir(self.save_translate_txt_path):
            os.makedirs(self.save_translate_txt_path)
        
        self.seg_text = seg_test()
        self.traslate_engine = M2M_translate()
        self.src_lang = "ja"
        self.target_lang = "zh"

    def load_data_path(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        image_paths = []
        image_name = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                _, extension = os.path.splitext(file)
                if extension.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
                    image_name.append(file.split(".jpg")[0])

        return image_paths, image_name
    
    def save_cropped_img(self, image_xyxy, scene, image_name):
        cnt = 0
        for i in image_xyxy:
            x1, y1, x2, y2 = i
            width = x2 - x1
            height = y2 - y1
            # Extract the region of interest (ROI) from the scene
            roi = scene[int(y1):int(y2), int(x1):int(x2)]
            # Save the ROI with the same size as the rectangle
            save_path = os.path.join(self.crop_textPath, f"{image_name}_{cnt}.jpg")
            cv2.imwrite(save_path, roi)
            # Increment the counter
            cnt += 1
        return

    def main(self):
        image_path, image_name = self.load_data_path()
        print("[INFO] Dealing with images: ", image_name)


        # { Crop image }
        cnt = 0
        for img_path in image_path:
            save_path = img_path.split(".")[0] + ".jpg"
            convert_to_jpg(img_path, save_path)

            if save_path.endswith("jpg"):
                image = cv2.imread(save_path)
                # croppedImageData = findSpeechBubbles(img_path, "complex", self.crop_textPath)              # OpenCV Version
                crpppedImageData, image_xyxy = self.seg_text.run(save_path)               # Roboflow
                self.save_cropped_img(image_xyxy, image, image_name[cnt])
                cnt += 1

        croppedImageList = os.listdir(self.crop_textPath)
        
        for crop_img in croppedImageList:   
            text_res = self.ocr_engine.text_generator(self.crop_textPath + "/" + crop_img)
            # print(text_res)
            translated_text = self.traslate_engine.run(text_res, self.src_lang, self.target_lang)

            with open(self.save_translate_txt_path+ crop_img.split(".jpg")[0] + ".txt", 'w', encoding='utf-8') as f:
                f.write(translated_text[0])

        return 
    
if __name__ == "__main__":
    # folder_path = "data/karrte/"
    folder_path = "data/testing_data/"
    test = action_manga(folder_path)
    test.main()