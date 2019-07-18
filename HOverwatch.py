import cv2
import numpy as np
from sys import exit
from time import time

folder_name = "assets"
haar_fullbody = cv2.CascadeClassifier(folder_name+'/haarcascade_fullbody.xml')
haar_upperbody = cv2.CascadeClassifier(folder_name+'/haarcascade_upperbody.xml')
haar_lowerbody = cv2.CascadeClassifier(folder_name+'/haarcascade_lowerbody.xml')
haar_face = cv2.CascadeClassifier(folder_name+'/haarcascade_frontalface.xml')
flir = cv2.VideoCapture(1)
webcam = cv2.VideoCapture("/home/braynyac/Videos/Walking_in_Spain.mp4")
webcam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
webcam.set(cv2.CAP_PROP_AUTOFOCUS, 0.25)
webcam.set(cv2.CAP_PROP_EXPOSURE, 0.2)
fgbg = cv2.createBackgroundSubtractorMOG2()

f_count = 0
k_sharp = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])


def corporal_detection(grayed_img):
    canvas = np.zeros(grayed_img.shape)
    uppers = haar_upperbody.detectMultiScale(grayed_img, scaleFactor=1.05)
    lowers = haar_lowerbody.detectMultiScale(grayed_img, scaleFactor=1.05)
    bodies = haar_fullbody.detectMultiScale(grayed_img, scaleFactor=1.05)
    faces = haar_face.detectMultiScale(grayed_img, scaleFactor=1.05)
    # upper_rects, _ = cv2.groupRectangles(uppers, 0, eps=0.2)
    # lower_rects, _ = cv2.groupRectangles(lowers, 0)

    for x, y, w, h in faces:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)
        cv2.rectangle(cam, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for x, y, w, h in uppers:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)
        cv2.rectangle(cam, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for x, y, w, h in lowers:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)
        cv2.rectangle(cam, (x, y), (x + w, y + h), (255, 0, 255), 2)

    for x, y, w, h in bodies:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)
        cv2.rectangle(cam, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # if len(lowers) and len(uppers) != 0:
    #     fullbody = haar_fullbody.detectMultiScale(grayed_img, 1.3, 2)
    #     full_rects, _ = cv2.groupRectangles(fullbody, 1)
    #     for x, y, w, h in full_rects:
    #         cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)
    #         cv2.rectangle(cam, (x, y), (x + w, y + h), 255, 2)

    return canvas


while True:
    t0 = time()
    therm = cv2.cvtColor(cv2.flip(flir.read()[1], 1), cv2.COLOR_BGR2GRAY)
    heatm = cv2.applyColorMap(therm, cv2.COLORMAP_JET)
    cam = cv2.flip(webcam.read()[1], 1)
    cam = cv2.resize(cam, (320, 240))
    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    corpus = corporal_detection(grayed_img=gray)
    # biblur = cv2.bilateralFilter(cam, 3, 75, 75)
    # fgmask = fgbg.apply(cam)
    # _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # cv2.drawContours(biblur, contours, -1, (0, 255, 255), -1)
    displaySize = (800, 600)
    heatm = cv2.resize(heatm, displaySize)

    # cv2.imshow("therm", therm)
    # cv2.imshow("THERMAL", heatm)
    cv2.imshow("Gray", cv2.resize(gray,displaySize))
    cv2.imshow("WebCam", cv2.resize(cam,displaySize))
    cv2.imshow("Corpus", cv2.resize(corpus,displaySize))
    # cv2.imshow("fgmask", fgmask)
    # cv2.imshow("THRESH", thresh)
    k = cv2.waitKey(3) & 0xFF
    if k == ord("q"):
        break

    # f_count += 1
    # if f_count == webcam.get(cv2.CAP_PROP_FRAME_COUNT):
    #     f_count = 0 #Or whatever as long as it is the same as next line
    #     webcam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print "FPS: ", (time()-t0)**-1


cv2.destroyAllWindows()
exit()
