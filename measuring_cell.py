import cv2
import numpy as np
import imutils


def count_cells(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_mop = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    D = cv2.distanceTransform(img_mop, cv2.DIST_L2, 5)
    localMax = np.zeros(D.shape, dtype=np.uint8)
    localMax[thresh == 255] = 255
    markers = cv2.connectedComponents(localMax)[1]
    markers = markers + 1
    markers[thresh == 0] = 0
    markers = cv2.watershed(image, markers)

    cell_count = len(np.unique(markers)) - 1

    for label in np.unique(markers):
        if label == -1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (255, 61, 139), 1, 5)
        cv2.putText(image, "{}".format(label), (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 155), 1)

    return image, cell_count


# Example usage:
image_path = './src/Raw-Data-BAT/HFD/II_2_1_HFD.tiff'
result_image, cell_count = count_cells(image_path)
print(result_image)
cv2.imshow("Result Image", result_image)
print("White Adipose Count: {} ".format(cell_count))
cv2.waitKey(0)
cv2.destroyAllWindows()
