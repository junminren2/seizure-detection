"""
This script is used to preprocess one-day videos
"""

import cv2
import os
import multiprocessing
from functools import partial
import time
import imageio
import configparser


def mul_preprocess(h_dir, hour, video_file):
        video_name = video_file.split('.')[0]
        video_path = os.path.join(h_dir, video_file)
        output_path = os.path.join(h_dir, "{}_{}.mp4".format(hour, video_name))

        reader = imageio.get_reader(video_path)
        writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'], **{'macro_block_size': 1})
        # writer = imageio.get_writer(output_path, fps=reader.get_meta_data()['fps'])
        for _, im in enumerate(reader):
            im = cv2.resize(im, (454, 256))
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            writer.append_data(eq_1)
        writer.close()
        os.remove(video_path)
        os.rename(output_path, video_path)

    # while True:
    #     # ret返回布尔量
    #     ret, frame = cap.read()
    #
    #     if ret == True:
    #       frame = cv2.resize(frame, (int(455), int(256)))
    #       # print('i={}'.format(i))
    #       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #       eq = cv2.equalizeHist(gray)
    #       eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    #       out.write(eq_1)
    #
    #       out.release()
    #       cap.release()
    #       os.remove(video_path)
    # os.rename(output_path, video_path)


def video_preprocess(kind_dir, mul_num=12):
    start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    days = os.listdir(kind_dir)
    for day in days:
     d_dir = os.path.join(kind_dir, day)
     hours = os.listdir(d_dir)


     for hour in hours:
      h_dir = os.path.join(d_dir, hour)
      minutes = os.listdir(h_dir)
      pool = multiprocessing.Pool(mul_num)
      func = partial(mul_preprocess, h_dir, hour)
      pool.map(func, minutes)
      pool.close()
      pool.join()
    end_time = time.time()
    print("Cost {} seconds.".format(end_time - start_time))



if __name__ == '__main__':
    start_time = time.time()
    pj_dir = os.path.dirname(os.path.realpath(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(pj_dir, 'config.ini'))
    kind_dir = config.get('Extract Move', 'dir')
    video_preprocess(kind_dir, 5)
    end_time = time.time()
    print("Cost {} seconds.".format(end_time - start_time))
