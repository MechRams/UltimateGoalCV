import cv2
import numpy as np, cv
from pipeline import GripPipeline

img = cv2.imread("img_rings/4.jpg")

pipeline = GripPipeline()

scale_percent = 20

#calculate the 50 percent of original dimensions
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

resized_img = cv2.resize(img, dsize)

while True:
    pipeline.process(resized_img)
    mask = pipeline.mask_output
    blobs = cv2.drawKeypoints(resized_img, pipeline.find_blobs_output, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    contours = cv2.drawContours(blobs, pipeline.find_contours_output, -1, (0, 255, 0), 2)
    cv2.imshow("result", contours)
    if cv2.waitKey(1) == 27:
        break