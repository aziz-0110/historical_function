import cv2
from skimage.feature import peak_local_max
# from skimage.morphology import watershed
from skimage.segmentation import watershed
from scipy import ndimage   # modul perhitungan scientific(integral numerik, diferensial, pemrosesan sinyal, dll)
"""Scipy.ndimage adalah subpaket dari perpustakaan SciPy yang menyediakan fungsi pemrosesan gambar multidimensi. 
Ini terutama digunakan untuk pemfilteran gambar, pengukuran, dan morfologi. Scipy.ndimage dapat digunakan untuk 
tugas-tugas seperti menghaluskan, mempertajam, mendeteksi tepi, dan mengurangi noise pada gambar. 
https://pieriantraining.com/a-beginners-guide-to-scipyndimage/ """
import numpy as np
import imutils

def controller(img_path):
    img = cv2.imread(img_path)

    title, gray = convert_grayscale(img)
    display_img(title, gray)

    title, thresh = thresholding(gray)
    display_img(title, thresh)

    title, morpho, markers, labels = morphological_opr(thresh)
    display_img(title, morpho)

    cont_cells(img, morpho, markers, labels)
    # title, cells = cont_cells(img, morpho, markers, labels)
    # display_img(title, cells)

    # cv2.destroyAllWindows()

def display_img(title, img):
    # memperkecil gambar
    resize = imutils.resize(img, width=850)

    cv2.imshow(title, resize)
    # cv2.imshow(title, cv2.resize(img, (1200, 600)))
    cv2.waitKey(0)
    return 0

def convert_grayscale(img):
    # konversi warna gambar jadi abu-abu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return "Grayscale", gray

def thresholding(gray):
    # konversi gambar abu-abu jadi biner
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return "Threshold", thresh


def morphological_opr(thresh):
    # membersihkan gambar biner atau menghilangkan noise pada objek
    img_mop = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)))
    img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))

    # perhitungan jarak transformasi
    D = ndimage.distance_transform_edt(img_mop)

    # mencari nalai lokal max di jarak tranformasi gambar menggunakan numpy
    local_max_coords = np.argwhere((D == ndimage.maximum_filter(D, size=20)) & (D > 0))

    # konversi kodinat lokal max ke bool
    localMax = np.zeros(D.shape, dtype=bool)
    localMax[tuple(local_max_coords.T)] = True

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    # untuk labeling gambar https://pyimagesearch.com/2015/11/02/watershed-opencv/
    labels = watershed(-D, markers, mask=img_mop)

    return "Distance Transform", D, markers, labels

def cont_cells(img, D, markers, labels):
    # perhtungan sel
    cell_count = len(np.unique(labels)) - 1
    print("Cell Count: {} ".format(cell_count))

    # menggambar lingkaran sel dan label
    for label in np.unique(labels):
        if label == 255:
            continue
        mask = np.zeros(D.shape, dtype="uint8")
        mask[labels == label] = 255

        # deteksi kontur objek https://learnopencv.com/contour-detection-using-opencv-python-c/
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # pengcekan kontur objek yg terdeteksi
        if cnts:
            # deteksi kontur dengan area maximum
            max_contour = max(cnts, key=cv2.contourArea)
            ((x,y), r) = cv2.minEnclosingCircle(max_contour)

            # menggambar linkaran dan label
            cv2.circle(img, (int(x), int(y)), int(r), (255, 61, 139), 3, 5)
            cv2.putText(img, "{}".format(label), (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 155), 1)

        display_img("Cell Count", img)
        # cv2.destroyAllWindows()
        # return "Cell Count", img
        # beri event untuk membersihkan jendela

img_path = './src/Raw-Data-BAT/HFD/II_2_1_HFD.tiff'
controller(img_path)
