import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
now_dir = os.getcwd()
sys.path.append(now_dir)

import json
from manga_ocr.manga_ocr.ocr import MangaOcr

class infer_ocr():
    def __init__(self):
        self.moce = MangaOcr()

    def text_generator(self, request, upload_dir):
        i = 1
        image_list = os.listdir(os.path.join( upload_dir,'cropped'))
        for img in image_list:
            i+=1
            text = self.mocr(os.path.join( upload_dir,'cropped',img))
            yield json.dumps({"id": i,"source":text})
        else:
            print("OCR complete!")
       