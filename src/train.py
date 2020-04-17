#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import sys
import numpy as np
import cv2

DIGIT_SIZE = 16

im = cv2.imread('../data/train.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(thresh,kernel,iterations = 1)

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, DIGIT_SIZE * DIGIT_SIZE), np.float32)
responses = []
keys = [i for i in range(48, 58)]

for cnt in contours:
    print(cv2.contourArea(cnt))
    if cv2.contourArea(cnt) > 100:
        [x, y, w, h] = cv2.boundingRect(cnt)
        print(h)

        if h > 25:

            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (DIGIT_SIZE, DIGIT_SIZE))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1, DIGIT_SIZE * DIGIT_SIZE))
                samples = np.append(samples, sample, 0)
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('norm', im)
                key = cv2.waitKey(100)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

samples = np.float32(samples)
responses = np.float32(responses)

cv2.imwrite("../data/train_result.png", im)
np.savetxt('../data/generalsamples.data', samples)
np.savetxt('../data/generalresponses.data', responses)
