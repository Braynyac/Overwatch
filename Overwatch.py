import cv2
import numpy as np
import sys
import time

flir = cv2.VideoCapture(1)
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture("C:\\Users\\Braynyac\\Videos\\HighSchoolSecCam360(1).mp4")
webcam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
webcam.set(cv2.CAP_PROP_AUTOFOCUS, 0.25)
webcam.set(cv2.CAP_PROP_EXPOSURE, 0.2)
fgbg = cv2.createBackgroundSubtractorMOG2()


def find_if_close(cnt1, cnt2, distance=7):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < distance:
                return True
            elif i == row1-1 and j == row2-1:
                return False


def unify_cont(contours):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))
    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH-1:
            for j, cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1, cnt2)
                if dist is True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i+1
    unified = []
    maximum = int(status.max()) + 1
    for i in xrange(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    return unified


def iter_gaussblur(img, iter, kernel):
    for i in xrange(iter):
        img = cv2.GaussianBlur(img, kernel, 0)
    return img


def iter_dilation(img, iter, kernel):
    for i in xrange(iter):
        img = cv2.dilate(img, kernel)
    return img


def simplified_conts(contours):
    approximations = []
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approximations.append(approx)
    return approximations


# f_count = 0
# k_sharp = np.array([[-1, -1, -1],
#                     [-1, 9, -1],
#                     [-1, -1, -1]])

while True:
    t0 = time.time()
    r = 2
    width = 80*r
    height = 60*r

    therm = cv2.cvtColor(flir.read()[1], cv2.COLOR_BGR2GRAY)
    therm = cv2.resize(therm, (width, height))
    therm = cv2.flip(therm, 1)
    heatm = cv2.applyColorMap(therm, cv2.COLORMAP_JET)
    cam = cv2.flip(webcam.read()[1], 1)
    area = width*height
    cam = cv2.resize(cam, (width, height))
    biblur = cv2.bilateralFilter(cam, 3, 75, 75)
    fgmask = fgbg.apply(cam)
    # blur = cv2.blur(fgmask, (3, 3))
    fgmask_blurred = iter_gaussblur(fgmask, 4, (9, 9))
    # fgmask_dilated = iter_dilation(fgmask_blurred, 3, (9, 9))
    _, blur_thresh = cv2.threshold(fgmask_blurred, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(blur_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = simplified_conts(contours)
    if len(contours) > 0:
        contours = unify_cont(contours)
    print "Contours:", len(contours)

    displaySize = (800, 600)

    motion_mask = np.zeros((height, width))
    cv2.drawContours(motion_mask, contours, -1, 255, -1)
    cv2.drawContours(cam, contours, -1, (0, 255, 255), 1)
    # print therm.shape
    # print motion_mask.shape
    merge = cv2.bitwise_and(therm, therm, mask=motion_mask.astype(dtype=np.uint8))
    merge = cv2.resize(merge, displaySize)
    motion_mask = cv2.resize(motion_mask, displaySize)
    therm = cv2.resize(therm, displaySize)
    heatm = cv2.resize(heatm, displaySize)
    cam = cv2.resize(cam, displaySize)
    biblur = cv2.resize(biblur, displaySize)
    cv2.imshow("therm", therm)
    cv2.imshow("THERMAL", heatm)
    cv2.imshow("WebCam", cam)
    # cv2.imshow("fgmask", fgmask)
    cv2.imshow("THRESH", blur_thresh)
    cv2.imshow("Motion Thermal", motion_mask)
    cv2.imshow("Merge", merge)
    k = cv2.waitKey(3) & 0xFF
    print "FPS: ", (time.time()-t0)**-1
    if k == ord("q"):
        break

cv2.destroyAllWindows()
sys.exit()