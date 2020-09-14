import cv2
import os
from PIL import Image
import torch
import argparse
import numpy as np
import sys
import time
import imageio
import decord as de

sys.path.append("D:\\videoAI\\RAFT-master\\core")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from collections import Counter

def convert2of(video_path, model_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))

    model = model.module
    model.to("cuda")
    model.eval()

    # path = os.path.join(video_path)
    subdirs = os.listdir(video_path)
    for subdir in subdirs:
        images = []

        # compute the path to the subdir
        subpath = os.path.join(video_path, subdir)
        # elems = os.listdir(subpath)
        # for elem in elems:
        #     # name = elem[:-4]
        #     path = os.path.join(subpath, elem)
        #     sample = os.listdir(path)
        #     for name in sample:
        #         print(name)
        #         ssspath = os.path.join(path, name)
        cap = cv2.VideoCapture(subpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
          ret, frame = cap.read()
          if ret:
           frame = frame[int(720 * 0.15):int(720 * 0.85), int(1280 * 0.15):int(1280 * 0.85)]
           frame = cv2.resize(frame, (int(455), int(256)))

           images.append(torch.from_numpy(frame).permute(2, 0, 1).float())
          else:
           break
    # cap = de.VideoReader(video_path, width = 455, height= 256)

    # fps = len(cap)
    # print(fps)
    # shape = cap[0].shape
    # print(shape)
    # i = 0
    # for i in cap:
    #  frame = cap[i].asnumpy()
    #  i = i + 1

        print("Read frames finished")
        images = torch.stack(images, dim=0)
        images = images.to("cuda")
        padder = InputPadder(images.shape)
        images = padder.pad(images)[0]
        fourcc = cv2.VideoWriter_fourcc(*'MP42')

        image1 = images[0, None]
        image2 = images[1, None]
        start_t = time.time()
        with torch.no_grad():
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print("Each prediction cost {}s".format(time.time() - start_t))
            output_image = viz(image1, flow_up)
        # out = cv2.VideoWriter(dst_path, fourcc,
        #                       fps, (output_image.shape[1], output_image.shape[0]))
        dst_path = os.path.join(video_path, 'move_opt')
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, subdir)
        out = imageio.get_writer(dst_path, format='mp4', mode='I', fps=fps)

        #print(output_image.shape)
        with torch.no_grad():
            for i in range(images.shape[0] - 1):

                image1 = images[i, None]
                image2 = images[i + 1, None]

                _, flow_up = model(image1, image2, iters=20, test_mode=True)
                tmp = viz(image1, flow_up)
                # tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
                # gray = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
                # tmp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # cv2.imshow('', tmp)
                # cv2.waitKey(1)

                # out.write(tmp)
                out.append_data(tmp)
        cap.release()
    # out.close()
    # out.release()

def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    #flo = np.concatenate([img, flo], axis=0)
    flo = np.concatenate([img, flo], axis=0)
    # shape = flo.shape
    # result = np.zeros(shape)
    # print(result)
    # for x in range(0, flo[:, , ]):
    #     for y in range(0, shape[1]):
    #         for z in range(0, shape[2]):
    #             if flo[x, y, z] <= 200:
    #                 result[x, y, z] = 0
    # flo = np.maximum(flo, 210)
    out = []
    a = 0

    # if flo[[255,255,255]] in flo:
    #     a = a+1
    #     print(a)
    # for i in range(250, 255):
    #     out.append(i)
    #     if out in flo:
    #      flo = 255
    #      flo = flo[:, :, [2, 1, 0]] / 255.0
    #     else:
    #      flo = flo[:, :, [2, 1, 0]] / 255.0
    # print('total_numebr:{}'.format(flo.shape[0]*flo.shape[1]))
    # # if [255,255,255] in flo:
    # #     a = a + 1
    # # flo.Counter([255,255,255])
    # print(sum(x.Counter([255,255,255]) for x in flo))
    # print(flo)
    # print((flo[:, :, [2, 1, 0]] / 255.0))
    return flo[:, :, [2, 1, 0]] / 255.0





def video2frames(video_path, dst_path):
    cap = cv2.VideoCapture(video_path)
    frame_cnt = 0
    frame_interval = 10
    interval = 0
    while frame_cnt < 10:
        _, frame = cap.read()
        interval += 1
        if interval > frame_interval:
            interval = 0
            frame = cv2.resize(frame, (int(1024*0.5), int(436*0.5)))
            im = Image.fromarray(frame)
            im.save(os.path.join(dst_path, "test_{}.png".format(frame_cnt)))
            frame_cnt += 1


def test_video(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    cv2.imshow('', frame)
    cv2.waitKey()


if __name__ == '__main__':
    # video2frames("/Users/jojen/Workspace/cityU/data/test/test.mp4",
    #              "/Users/jojen/Workspace/cityU/data/test/test_raft")
    start_time = time.time()
    convert2of("E:\\data_YU_20200606\\09_09_case_dataset\\move",
               "D:\\videoAI\\RAFT-master\\models\\raft-things.pth")
    end_time = time.time()
    print('used_time = {}s'.format(end_time-start_time))
    # test_video("E:\\data_YU_20200606\\09_09_case_dataset\\test_of.avi")