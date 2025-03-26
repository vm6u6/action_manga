import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from inpaint_modules.inpaint import INPAINTERS

def debug_img(img, mask, result):

    # result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(output_path, result_bgr)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Orignal")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Result")
    plt.imshow(result)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return 

class infer_inpaint():
    def __init__(self):
        self.inpainter = INPAINTERS.get('lama_large_512px')()
        self.inpainter.load_model()
    
    def run(self, image, image_xyxy, margin):
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for xyxy in image_xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.shape[1], x2 + margin)
            y2 = min(image.shape[0], y2 + margin)
            full_mask[y1:y2, x1:x2] = 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            result = self.inpainter.inpaint(image, full_mask)
        except Exception as e:
                    print(f"Inpaint failed: please check inpaint_img.py {e}")
    
        debug_img(image, full_mask, result)
        return result


def test_inpaint_image_with_class(image_path, mask_path, output_path=None):
    inpainter = INPAINTERS.get('lama_large_512px')()
    
    inpainter.load_model()
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    result = inpainter.inpaint(img, mask)

    if output_path:
        debug_img(img, mask, result)
    
    return result

# 使用示例
if __name__ == "__main__":
    image_path = "data/testing_data/fll_test2.jpg"  
    mask_path = "random_mask.png"    
    output_path = "inpainted_result.jpg" 
    result = test_inpaint_image_with_class(image_path, mask_path, output_path)