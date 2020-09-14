import cv2
import getopt
import os, sys
import imutils
import shutil
import numpy as np
import time
import multiprocessing

def rename(day_dir, exper, Num):
    prefix, day = os.path.split(day_dir)
    elems = os.listdir(day_dir)
    for elem in elems:
        h_dir = os.path.join(day_dir, elem)
        sample = os.listdir(h_dir)
        for name in sample:
            v_path = os.path.join(h_dir, name)
            os.rename(v_path, os.path.join(h_dir, exper + '_' + Num + '_' + day + '_' + elem + '_' + name))
def flow(prvs, next, hsv):
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 9, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def op_flow(day_dir):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h_dirs = os.listdir(day_dir)
    out_dir = os.path.join(day_dir, 'op_flow')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for h in h_dirs:
        h_dir = os.path.join(day_dir, h)
        sample = os.listdir(h_dir)
        for name in sample:
            v_path = os.path.join(h_dir, name)
            cap = cv2.VideoCapture(v_path)
            o_path = os.path.join(out_dir, name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(o_path, fourcc, fps, (frame_width, frame_height))
            a, b1 = cap.read()
            while a == False:
                a, b1 = cap.read()
            frame1 = b1.copy()
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            while True:

                a, b2 = cap.read()
                if a:
                    hsv = np.zeros_like(b1)
                    hsv[..., 1] = 255

                    frame2 = b2.copy()
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                    op = flow(gray1, gray2, hsv)

                    out.write(op)
                    gray1 = gray2

                else:
                    break

            cap.release()
            out.release()



opts, args = getopt.getopt(sys.argv[1:], '-h-d:-e:-n:')
day_dir = ''
num = ''
exper = ''
pj_dir = os.getcwd()
for opt_name, opt_value in opts:
    if opt_name == '-d':
        day_dir = opt_value
    if opt_name == '-e':
        exper = opt_value
    if opt_name == '-n':
        num = opt_value

if day_dir == '':
    print('pleas input video directory for optical flow.')
    sys.exit()
st = time.time()
rename(day_dir, exper, num)

op_flow(day_dir)
et = time.time()
used_t = (et-st)/60
print('Prediction compeleted')
print('Used time :' + str(used_t) + 'min')
