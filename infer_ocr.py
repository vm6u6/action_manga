import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
now_dir = os.getcwd()
sys.path.append(now_dir)

import json
from manga_ocr.manga_ocr.ocr import MangaOcr

class infer_ocr():
    def __init__(self):
        self.mocr = MangaOcr()

    def text_generator(self, img_dir):
        text = self.mocr(img_dir)
        print(text)
        return text

       