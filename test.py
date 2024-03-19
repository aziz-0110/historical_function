import cv2
from skimage.feature import peak_local_max
# from skimage.morphology import watershed
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils


def count_cells(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display grayscale image
    cv2.imshow("Grayscale", gray)
    cv2.waitKey(0)

    # Thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Display thresholded image
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(0)

    # Morphological operations
    img_mop = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # Distance transform
    D = ndimage.distance_transform_edt(img_mop)

    # Find local maxima
    # Find local maxima in distance transform image using numpy
    local_max_coords = np.argwhere((D == ndimage.maximum_filter(D, size=20)) & (D > 0))

    # Convert local maxima coordinates to boolean mask
    localMax = np.zeros(D.shape, dtype=bool)
    localMax[tuple(local_max_coords.T)] = True

    # Display distance transform image
    cv2.imshow("Distance Transform", D)
    cv2.waitKey(0)

    # Label connected components
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    # Watershed segmentation
    labels = watershed(-D, markers, mask=img_mop)

    # Count cells
    cell_count = len(np.unique(labels)) - 1
    print("Cell Count: {} ".format(cell_count))

    # Draw circles around cells and label them
    for label in np.unique(labels):
        if label == 255:
            continue
        mask = np.zeros(D.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours are found
        if cnts:
            # Find the contour with the maximum area
            max_contour = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(max_contour)
            # Draw the circle and label
            cv2.circle(image, (int(x), int(y)), int(r), (255, 61, 139), 3, 5)
            cv2.putText(image, "{}".format(label), (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 155), 1)

        # c = max(cnts, key=cv2.contourArea)
        # ((x, y), r) = cv2.minEnclosingCircle(c)
        # cv2.putText(image, "{}".format(label), (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 155), 1)

        # Display result
        cv2.imshow("Cell Count", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage:
image_path = './src/Raw-Data-BAT/HFD/II_2_1_HFD.tiff'
count_cells(image_path)
