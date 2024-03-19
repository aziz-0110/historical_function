import cv2
from skimage.feature import peak_local_max
# from skimage.morphology import watershed
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np

def read_image(image_path):
    """
    Read an image from the specified path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Image data.
    """
    image = cv2.imread(image_path)
    display_image(image, "Original Image")
    return image

def convert_to_grayscale(image):
    """
    Convert the input image to grayscale.

    Args:
        image (np.ndarray): Input image data.

    Returns:
        np.ndarray: Grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image(gray, "Grayscale Image")
    return gray

def display_image(image, title):
    """
    Display the input image with the specified title.

    Args:
        image (np.ndarray): Image data.
        title (str): Title for the displayed window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)

def apply_threshold(image):
    """
    Apply Otsu's thresholding to the input image.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Thresholded image.
    """
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    display_image(thresh, "Thresholded Image")
    return thresh

def apply_morphological_operations(image):
    """
    Apply morphological operations to the input image.

    Args:
        image (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Image after morphological operations.
    """
    img_mop = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    display_image(img_mop, "Morphological Operations")
    return img_mop

def compute_distance_transform(image):
    """
    Compute the distance transform of the input binary image.

    Args:
        image (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Distance transform image.
    """
    D = ndimage.distance_transform_edt(image)
    display_image(D, "Distance Transform")
    return D

def find_local_maxima(distance_transform):
    """
    Find local maxima in the distance transform image.

    Args:
        distance_transform (np.ndarray): Distance transform image.

    Returns:
        np.ndarray: Binary mask indicating the local maxima.
    """
    local_max_coords = np.argwhere((distance_transform == ndimage.maximum_filter(distance_transform, size=20)) & (distance_transform > 0))
    localMax = np.zeros(distance_transform.shape, dtype=bool)
    localMax[tuple(local_max_coords.T)] = True
    return localMax

def label_connected_components(local_maxima):
    """
    Label connected components in the input binary image.

    Args:
        local_maxima (np.ndarray): Binary mask indicating local maxima.

    Returns:
        np.ndarray: Labeled image.
    """
    markers = ndimage.label(local_maxima, structure=np.ones((3, 3)))[0]
    display_image(markers, "Connected Components")
    return markers

def perform_watershed_segmentation(distance_transform, markers, mask):
    """
    Perform watershed segmentation.

    Args:
        distance_transform (np.ndarray): Distance transform image.
        markers (np.ndarray): Labeled image.
        mask (np.ndarray): Mask image.

    Returns:
        np.ndarray: Segmented image.
    """
    labels = watershed(-distance_transform, markers, mask=mask)
    display_image(labels, "Segmented Image")
    return labels

def draw_cells(image, labels):
    """
    Draw circles around cells and label them.

    Args:
        image (np.ndarray): Input image data.
        labels (np.ndarray): Segmented image.
    """
    for label in np.unique(labels):
        if label == 255:
            continue
        mask = np.zeros(labels.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            max_contour = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(max_contour)
            cv2.circle(image, (int(x), int(y)), int(r), (255, 61, 139), 3, 5)
            cv2.putText(image, "{}".format(label), (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 155), 1)
    display_image(image, "Cell Count")
    cv2.destroyAllWindows()

