# coding=utf-8
import numpy as np
import cv2
import math
from PIL import Image


def deskew(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    erode_Img = cv2.erode(gray, kernel)
    eroDil = cv2.dilate(erode_Img, kernel)  # erode and dilate

    canny = cv2.Canny(eroDil, 50, 150)  # edge detection

    lines = cv2.HoughLinesP(
        canny, 0.8, np.pi / 180, 90, minLineLength=100, maxLineGap=10
    )  # Hough Lines Transform
    drawing = np.zeros(src.shape[:], dtype=np.uint8)

    maxY = 0
    degree_of_bottomline = 0
    index = 0
    if lines is None or len(lines) == 0:
        print("No text found.")
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        if (x1 - x2) != 0:
            k = float(y1 - y2) / (x1 - x2)
            degree = np.degrees(math.atan(k))
            if index == 0:
                maxY = y1
                degree_of_bottomline = (
                    degree  # take the degree of the line at the bottom
                )
            else:
                if y1 > maxY:
                    maxY = y1
                    degree_of_bottomline = degree
            index = index + 1

    img = Image.fromarray(src)
    rotateImg = img.rotate(degree_of_bottomline)
    rotateImg_cv = np.array(rotateImg)

    return rotateImg_cv


if __name__ == "__main__":
    deskew()
