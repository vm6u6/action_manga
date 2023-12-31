import cv2
import numpy as np

def findSpeechBubbles(imagePath, method, upload_dir):
    # read image
    image = cv2.imread(imagePath)
    # gray scale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # filter noise
    imageGrayBlur = cv2.GaussianBlur(imageGray, (3, 3), 0)
    if method != 'simple':
        # recognizes more complex bubble shapes
        imageGrayBlurCanny = cv2.Canny(imageGrayBlur, 50, 500)
        binary = cv2.threshold(imageGrayBlurCanny, 235,
                               255, cv2.THRESH_BINARY)[1]
    else:
        # recognizes only rectangular bubbles
        binary = cv2.threshold(imageGrayBlur, 235, 255, cv2.THRESH_BINARY)[1]

    # find contours
    contours = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # get the list of cropped speech bubbles

    croppedImageList = []
    i = 0
    for contour in contours:

        contour = contour.astype(np.int32)
        rect = cv2.boundingRect(contour)
        [x, y, w, h] = rect

        # filter out speech bubble candidates with unreasonable size
        if w < 500 and w > 40 and h < 500 and h > 40:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            croppedImage = image[y:y+h, x:x+w]
            croppedImageList.append(croppedImage)
            cv2.imwrite(upload_dir+"/"+ str(i)+".jpg", croppedImage)
            i += 1

    return croppedImageList