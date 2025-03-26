from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import os

class pannel_infer():
    def __init__(self):
        rf = Roboflow(api_key="62xEJQQ4CxpSL7edYbnx")
        project = rf.workspace().project("manga-segment")
        self.model = project.version(15).model

    def add_black_border(self, image, border_size=3):
        bordered_image = cv2.copyMakeBorder(
            image, 
            border_size, 
            border_size, 
            border_size, 
            border_size, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0] 
        )
        return bordered_image

    def save_panels(self, image, detections, img_basename, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        panel_paths = []
        for i, bbox in enumerate(detections):
            x1, y1, x2, y2 = map(int, bbox)
            panel = image[y1:y2, x1:x2]
            output_path = os.path.join(output_dir, f"{img_basename}_panel_{i}.jpg")
            cv2.imwrite(output_path, panel)
            panel_paths.append(output_path)
            
        return panel_paths

    def run(self, img_path):
        img = cv2.imread(img_path)

        # img_with_border = self.add_black_border(img)
        # bordered_img_path = img_path.replace('.jpg', '_bordered.jpg')
        # cv2.imwrite(bordered_img_path, img_with_border)

        result = self.model.predict(img_path, confidence=50).json()         # .jpg
        labels = [item["class"] for item in result["predictions"]]

        detections = sv.Detections.from_inference(result)
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        
        annotated_image = mask_annotator.annotate(
            scene=img, detections=detections)
        # annotated_image = label_annotator.annotate(
        #     scene=annotated_image, detections=detections, labels=labels)
        # sv.plot_image(image=annotated_image, size=(16, 16))
        
        # os.remove(bordered_img_path)
        # panel_paths = self.save_panels(img, detections.xyxy, output_dir)
        return annotated_image, detections.xyxy
    
if __name__ == "__main__":
    tmp = pannel_infer()
    test_img_path = "./data/testing_data/jjtk_test.jpg"
    annotated_image, xyxy = tmp.run(test_img_path)