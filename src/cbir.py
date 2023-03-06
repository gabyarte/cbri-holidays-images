import cv2
import glob


from sift import sift_euclidean
from smart_histogram import smart_histogram_chisqr

images = [cv2.imread(file) for file in glob.glob('../data/*.jpg')]

image = cv2.imread('../data/131502.jpg')

images.sort(key=lambda img: smart_histogram_chisqr(image, img))
smart_histogram_list = images[:10]

images.sort(key=lambda img: sift_euclidean(image, img))
sift_list = images[:10]

print('Smart-Hist:')
