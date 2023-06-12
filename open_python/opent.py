import cv2
import numpy as np

from matplotlib import pyplot as plt


img = cv2.imread("shapes.png")

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, img_threshold = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

for contour in contours:
    if i == 0:
        i = 1
        continue

    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # Hien thi contour
    # cv2.drawContours(img, [contour], 0, (0, 0, 250), 8)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

    if len(approx) == 3:
        cv2.putText(
            img, "Tam giac", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 2
        )

    elif len(approx) == 4:
        cv2.putText(
            img, "Tu giac", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    elif len(approx) == 5:
        cv2.putText(
            img, "Ngu giac", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    elif len(approx) == 6:
        cv2.putText(
            img, "Luc giac", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    else:
        cv2.putText(
            img, "Tron", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

cv2.imshow("window", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
