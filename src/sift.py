import cv2

img = cv2.imread('../data/100000.jpg')
img2 = cv2.imread('../data/100200.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()

matches = bf.match(des1, des2)

sum(c.distance for c in matches)
