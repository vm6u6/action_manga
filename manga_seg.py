from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

class seg_test():
    def __init__(self):
        self.rf = Roboflow(api_key="g2d6XQ9NG01K5dVIhSFL")
        self.project = self.rf.workspace().project("manga-translator-segmentation")
        self.model = self.project.version(10).model


    def run(self, img_path):
        result = self.model.predict(img_path, confidence=40).json()
        labels = [item["class"] for item in result["predictions"]]

        detections = sv.Detections.from_roboflow(result)


        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        image = cv2.imread(img_path)

        annotated_image = mask_annotator.annotate(
            scene=image, detections=detections)
        
        # annotated_image = label_annotator.annotate(
        #     scene=annotated_image, detections=detections, labels=labels)
        # print(detections.shape)


        # sv.plot_image(image=annotated_image, size=(16, 16))
        # cv2.imwrite(save_path, annotated_image)
        return annotated_image, detections.xyxy

if __name__ == "__main__":
    test = seg_test()
    img_path = "/home/user/action_manga/data/karrte/01.jpg"
    save_path = "./seg_test.jpg"
    test.run(img_path, save_path)