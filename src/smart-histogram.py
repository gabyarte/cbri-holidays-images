import cv2


def equalize(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE()
    hsv_img[..., 2] = clahe.apply(hsv_img[..., 2])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)


def split_image(img):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    topLeft = img[0:cY, 0:cX]
    topRight = img[0:cY, cX:w]
    bottomLeft = img[cY:h, 0:cX]
    bottomRight = img[cY:h, cX:w]
    return [topLeft, topRight, bottomLeft, bottomRight]


def generate_histograms(img, histSize=256, histRange=(0, 256), accumulate=False):
    bgr_img = cv2.split(img)
    b_hist = cv2.calcHist(bgr_img, [0], None, [
                          histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_img, [1], None, [
                          histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_img, [2], None, [
                          histSize], histRange, accumulate=accumulate)
    return [b_hist, g_hist, r_hist]


def smart_histogram_descriptor(img):
    equalized = equalize(img)
    split = split_image(equalized)
    return list(map(generate_histograms, split))


def distance(img1, img2):
    distance = 0
    for im1, im2 in zip(img1, img2):
        for hist1, hist2 in zip(im1, im2):
            distance += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return distance


img = cv2.imread('../data/100000.jpg')
im1 = smart_histogram_descriptor(img)


img2 = cv2.imread('../data/131502.jpg')
im2 = smart_histogram_descriptor(img2)


print(distance(im1, im2))
