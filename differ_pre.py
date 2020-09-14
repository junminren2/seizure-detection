import cv2
import os
import time
import
def diff_pre(input_dir, out_dir):
        time_start = time.time()
        elems = os.listdir(input_dir)
        for elem in elems:
            name = elem[:-4]
            path = os.path.join(input_dir, elem)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.set(cv2.CAP_PROP_POS_FRAMES,1)
            a, b1 = cap.read()
            cv2.imshow('b1', b1)
            cap.set(cv2.CAP_PROP_POS_FRAMES,num_fps)
            a, b2 = cap.read()
            cv2.imshow('b2', b2)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            o_path = os.path.join(out_dir, name + '.mp4')
            out = cv2.VideoWriter(o_path, fourcc, fps, (frame_width, frame_height))

            while (True):
                ret, frame = cap.read()
                if ret == True:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    eq = cv2.equalizeHist(gray)
                    eq_1 = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
                    out.write(eq_1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            out.release()
            time_end = time.time()
            print('time cost', time_end - time_start, 's')
def find_threshold():
    diffs = []
    global cap, point1, point2
    find_timepos()
    min_x = min(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    width = abs(point1[0] - point2[0])
    height = abs(point1[1] - point2[1])
    #print("compute diff")
    while cap.isOpened():
        #print("computing......")
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        #print(ret1)
        #print(ret2)
        if ret1 and ret2:
            frame1 = frame1.copy()
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray1[min_y:min_y + height, min_x:min_x + width] = 0
            frame2 = frame2.copy()
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            gray2[min_y:min_y + height, min_x:min_x + width] = 0
            diff = cv2.absdiff(gray1, gray2)
            sum_diff = sum(diff)
            sum_diff = sum(sum_diff)
            diffs.append(sum_diff)
        else:
            print("file broken")
            break
    cap.release()
    cv2.destroyAllWindows()
    print('个数：', len(diffs))
    print('平均值:', np.mean(diffs))
    print('中位数:', np.median(diffs))


input_dir = ''
diff_pre(input_dir, out_dir)