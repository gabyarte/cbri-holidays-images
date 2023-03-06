import cv2


def sift_euclidean(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    return sum(c.distance for c in matches)


img = cv2.imread('../data/108102.jpg')
img2 = cv2.imread('../data/107203.jpg')
sift_euclidean(img, img2)
