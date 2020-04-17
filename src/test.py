#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import cv2
import numpy as np

DIGIT_SIZE = 16

#######   training part    ###############
samples = np.loadtxt('../data/generalsamples.data', np.float32)
responses = np.loadtxt('../data/generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

############################# testing part  #########################

im = cv2.imread('../data/test.png')
out = np.zeros(im.shape, np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(thresh,kernel,iterations = 1)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 35:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (DIGIT_SIZE, DIGIT_SIZE))
            roismall = roismall.reshape((1, DIGIT_SIZE*DIGIT_SIZE))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=5)
            string = str(int((results[0][0])))
            cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

cv2.imshow('im', im)
cv2.imshow('out', out)
cv2.waitKey(0)
