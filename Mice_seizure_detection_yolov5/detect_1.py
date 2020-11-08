import argparse
import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import ( non_max_suppression,
                           scale_coords, xyxy2xywh, set_logging)
from utils.torch_utils import select_device, load_classifier

#img: load tensor, tensor.Size = ([3, 384, 640])
#im0: original data
def detect(img, im0):
    prob = 0

    weights, = \
        opt.weights

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # half precision only supported on CUDA
    half = device.type != 'cpu'

    # Load model
    # load FP32 model
    model = attempt_load(weights, map_location=device)
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
    # Run inference
        img[0] = img[0] - 144.7748
        img[1] = img[1] - 107.7354
        img[2] = img[2] - 99.4750
        img = torch.from_numpy(img).to(device)
        print(img.shape)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if n == 1:
                        prob += det.cpu().numpy()[0][4]
                        print(prob)
                    elif n > 1:
                        a = []
                        b = []
                        for h in range(n):
                            a1 = det.cpu().numpy()[h][4]
                            a.append(a1)  # probability
                            b.append(h)  # index of different detections
                        max_a = a.index(max(a))
                        del b[max_a]
                        det = np.delete(det.cpu().numpy(), b, 0)
                        prob += det[0][4]
                        det = torch.tensor(det)

                # Write results
                det = torch.tensor(det)
                for *xyxy, conf, cls in reversed(det):
                    # global x_center, y_center
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        x_center = xywh[0]
                        y_center = xywh[1]
                        return x_center, y_center


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/exp9/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()

