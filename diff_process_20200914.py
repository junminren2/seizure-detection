import cv2
import os
import time
import numpy as np
import imutils
import shutil
def rename(input_dir, exper, Num):
  path = os.path.join(input_dir)
  subdirs = os.listdir(path)
  for subdir in subdirs:
    # compute the path to the subdir
    subpath = os.path.join(input_dir, subdir)
    elems = os.listdir(subpath)
    for elem in elems:
      # name = elem[:-4]
      sspath = os.path.join(subpath, elem)
      sample = os.listdir(sspath)
      for name in sample:
        ssspath = os.path.join(sspath, name)
        os.rename(ssspath, sspath + '//' + exper + '_' + Num + '_' + subdir + '_' + elem + '_' + name)
def extract(input_dir, out_dir):
        #diffs = []
        i = 0
        j = 0
        time_start = time.time()
        path = os.path.join(input_dir)
        subdirs = os.listdir(path)
        for subdir in subdirs:

          # compute the path to the subdir
          subpath = os.path.join(input_dir, subdir)
          elems = os.listdir(subpath)
          for elem in elems:
              # print(h)
              h_dir = os.path.join(subpath, elem)
              sample = os.listdir(h_dir)
              for name in sample:

                  v_path = os.path.join(h_dir, name)
                  cap = cv2.VideoCapture(v_path)
                  num_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                  cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

                  a, b1 = cap.read()
                  if a == False:
                      continue
                  cap.set(cv2.CAP_PROP_POS_FRAMES, int((num_fps - 1 - 1) / 2))

                  a, b2 = cap.read()
                  if a == False:
                      continue

                  cap.set(cv2.CAP_PROP_POS_FRAMES, num_fps - 1)
                  a, b3 = cap.read()
                  if a == False:
                      continue

                  frame1 = b1.copy()

                  frame1 = frame1[128:720, 0:1280]
                  frame1 = imutils.resize(frame1, width=500)
                  gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                  gray1 = cv2.equalizeHist(gray1)
                  frame2 = b2.copy()
                  frame2 = frame2[128:720, 0:1280]
                  frame2 = imutils.resize(frame2, width=500)
                  gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                  gray2 = cv2.equalizeHist(gray2)
                  frame3 = b3.copy()
                  frame3 = frame3[128:720, 0:1280]
                  frame3 = imutils.resize(frame3, width=500)
                  gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
                  gray3 = cv2.equalizeHist(gray3)
                  diff = cv2.absdiff(gray1, gray2)
                  diff1 = cv2.medianBlur(diff, 3)
                  diff2 = cv2.absdiff(gray2, gray3)
                  diff3 = cv2.medianBlur(diff2, 3)

                  thresh = cv2.threshold(diff1, 40, 255, cv2.THRESH_BINARY)[1]

                  sum_diff = cv2.countNonZero(thresh)

                  thresh1 = cv2.threshold(diff3, 40, 255, cv2.THRESH_BINARY)[1]
                  sum_diff_1 = cv2.countNonZero(thresh1)

                  sum_diff_2 = sum_diff + sum_diff_1
                  if sum_diff_2 > 1000:
                      shutil.copy(v_path, out_dir)
                  # print(v_path)
                  # print('value_diff_f1f2: %s' % sum_diff_2)
                  # print('value_diff_f2f3: %s' % sum_diff_1)
                  cap.release()
              shutil.rmtree(h_dir, ignore_errors=True)
exper = 'AL' #experimentor
Num = '220' #kind of mice
input_dir = 'D:\\C10\\case\\test' # input folder for top layer of one mice
out_dir = 'D:\\C10\\case\\test\\extract' #build a new folder
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
#rename(input_dir, exper, Num)
extract(input_dir, out_dir)
