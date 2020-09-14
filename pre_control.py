import numpy as np
import os
import shutil
import xlrd
import cv2
import config
import random
import time
def preprocess_control(input_dir_1, output_dir_1):
    time_start = time.time()
    #k = 150  # frame interval
    blockSize = 16

    v_l = os.listdir(input_dir_1)
    for v in v_l:
        k = random.randint(128, 192)
        name = v[:-4]
        v_path = os.path.join(input_dir_1, v)
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        i = 0
        j = int(i / k) + 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        o_path = os.path.join(output_dir_1, name + '_' + str(j) + '.mp4')
        out = cv2.VideoWriter(o_path, fourcc, fps, (frame_width, frame_height))

        while (True):
            ret, frame = cap.read()  ##ret返回布尔量

            if ret == True:
                i += 1
                if (i % k != 0):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    eq = cv2.equalizeHist(gray)
                    eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
                    out.write(eq_1)
                else:
                    j = int(i / k) + 1
                    out.release()
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    o_path = os.path.join(output_dir_1, name + '_' + str(j) + '.mp4')
                    out = cv2.VideoWriter(o_path, fourcc, fps, (frame_width, frame_height))
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
def main():
    time_start = time.time()
    newnewdir = 'E:\\alice_video\\257\\2020-06-03\\00\\test' #control files are moved in this direcitionary

    input_dir_1 = newnewdir
    output_dir_1 = 'E:\\alice_video\\257\\2020-06-03\\00\\test' #preprocessed case files are moved in this direcitionary
    #if not os.path.exists(output_dir_1):
    #     os.makedirs(output_dir_1)
    preprocess_control(input_dir_1, output_dir_1)
    print('preprocess_control')
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
if __name__ == "__main__":
    main()