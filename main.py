
import os, sys

import numpy as np
from infer_ocr import infer_ocr
from capture_text import findSpeechBubbles
from manga_seg import seg_test
from utils import convert_to_jpg
import cv2
from translate import M2M_translate, LLM_translate
from inpaint_img import infer_inpaint
from pannel_dectect import pannel_infer

class action_manga():
    def __init__(self, folder_path):
        self.ocr_engine = infer_ocr()
        self.inpaint_engine = infer_inpaint()


        self.inpaint_img_path = folder_path + "/inpainted/"
        if not os.path.isdir(self.inpaint_img_path):
            os.makedirs(self.inpaint_img_path)

        self.folder_path = folder_path
        translator = "LLM"
        self.crop_textPath = folder_path + "/cropped/"
        if not os.path.isdir(self.crop_textPath):
            os.makedirs(self.crop_textPath)

        self.save_translate_txt_path = folder_path + "/translated/"
        if not os.path.isdir(self.save_translate_txt_path):
            os.makedirs(self.save_translate_txt_path)
        
        self.seg_text = seg_test()
        if translator == "m2m":
            self.translate_engine = M2M_translate()
            self.src_lang = "ja"
            self.target_lang = "zh"
        elif translator == "LLM":
            self.translate_engine = LLM_translate(        
                api_config="LLM.txt",
                max_rpm=3,
                model_name="gemini-2.0-pro-exp-02-05",
                provider="Google")

        self.pennel_engine = pannel_infer() 
        self.save_pannel_path = folder_path + "/pannel/"
        if not os.path.isdir(self.save_pannel_path):
            os.makedirs(self.save_pannel_path)


    def load_data_path(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        image_paths = []
        image_name = []

        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)
            
            if os.path.isfile(file_path):
                _, extension = os.path.splitext(file)
                if extension.lower() in image_extensions:
                    image_paths.append(file_path)
                    image_name.append(file.split(".jpg")[0])
                    
        return image_paths, image_name
    
    def save_cropped_img(self, image_xyxy, scene, image_name):
        cnt = 0
        for i in image_xyxy:
            x1, y1, x2, y2 = i
            width = x2 - x1
            height = y2 - y1
            roi = scene[int(y1):int(y2), int(x1):int(x2)]
            save_path = os.path.join(self.crop_textPath, f"{image_name}_{cnt}.jpg")
            cv2.imwrite(save_path, roi)
            cnt += 1
        return

    def main(self):
        image_path, image_name = self.load_data_path()
        print("[INFO] Dealing with images: ", image_name)

        # ### { Crop image OCR / Inpainting } ###
        # cnt = 0
        # image_xyxy_list = []
        # for img_path in image_path:
        #     save_path = img_path.split(".")[0] + ".jpg"
        #     convert_to_jpg(img_path, save_path)

        #     if save_path.endswith("jpg"):
        #         image = cv2.imread(save_path)
        #         # croppedImageData = findSpeechBubbles(img_path, "complex", self.crop_textPath)                     # OpenCV Version (Not recommend)
        #         crpppedImageData, image_xyxy = self.seg_text.run(save_path)                                         # Roboflow
        #         self.save_cropped_img(image_xyxy, image, image_name[cnt])
        #         image_xyxy_list.append(image_xyxy)

        #         inpainted_image = self.inpaint_engine.run(image, image_xyxy, margin=5)
        #         inpainted_save_path = self.inpaint_img_path + image_name[cnt] + "_inpainted.jpg"
        #         cv2.imwrite(inpainted_save_path, cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))
        #         print(f"Save inpaint img to : {inpainted_save_path}")

        #         cnt += 1
        # croppedImageList = os.listdir(self.crop_textPath)

        # ### { Translate } ###
        # img_text = []
        # for crop_img in croppedImageList:   
        #     text_res = self.ocr_engine.text_generator(self.crop_textPath + "/" + crop_img)
        #     # translated_text = self.translate_engine.run(text_res, self.src_lang, self.target_lang)                # m2m
        #     # translated_text = self.translate_engine.translate_text(text_res)                                      # Gemini 有次數限制
        #     img_text.append(text_res)

        #     ### [TODO] length limited with gemini

        # translated_text_list = self.translate_engine.translate_txt_list(img_text)
        # for i in range( len(croppedImageList) ):  
        #     crop_img = croppedImageList[i]
        #     translated_text = translated_text_list[i]
        #     with open(self.save_translate_txt_path+ crop_img.split(".jpg")[0] + ".txt", 'w', encoding='utf-8') as f:
        #         f.write(translated_text)


        ### { Animate } ###
        inpainted_image_list = os.listdir(self.inpaint_img_path)
        for inpainted_img in inpainted_image_list:
            in_img_path = self.inpaint_img_path + inpainted_img 
            annotated_image, xyxy = self.pennel_engine.run(in_img_path)
            pannel_paths = self.pennel_engine.save_panels(annotated_image, xyxy, inpainted_img, self.save_pannel_path)


            
        # Diffision


        ### { Refilled } ###



        return 
    
if __name__ == "__main__":
    # folder_path = "data/karrte/"
    folder_path = "data/testing_data/"
    test = action_manga(folder_path)
    test.main()