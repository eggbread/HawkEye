from __future__ import division
import time
import torch
from torch.autograd import Variable
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pickle as pkl
import argparse
from sort import *
from test import *
from SORTCode import sort as sort2
import numpy as np

class HawkEye():
    def __init__(self):
        self.y_axios = -1
        self.clickFlag = False

    def prep_image(self, img, inp_dim):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """

        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim


    def write(self, x,  img):
        x = torch.tensor(x)
        c1 = tuple(x[0:2].int())
        c2 = tuple(x[2:4].int())
        cls = int(x[-3])
        label = "{0}".format(classes[cls])
        label += " " + str(int(x[4].item())) + " "
        size = x[-2].item()
        if size == 0:
            return;
        label += str(size) + " "
        if x[-1] == 1:
            label += "come"
        elif x[-1] == 2:
            label += "warning"
        color = colors[((cls+1) * int(x[4])) % len(colors)]
        cv2.rectangle(img, c1, c2, color, 3)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
        return img

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.y_axios = y
            self.clickFlag = True

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights("../../yolov3.weights")
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    cap = cv2.VideoCapture("../../flyingBirds.mp4")

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    mot_tracker = Sort(frame_width)
    mot_tracker2 = sort2.Sort()
    hawkEye = HawkEye()
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', hawkEye.mouse_callback)
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
    test = test()
    while cap.isOpened():
        if frames % 3 != 0:
            frames += 1
            continue
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:

            img, orig_im, dim = hawkEye.prep_image(frame,inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            if output is not None:
                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

                output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

                output[:, 1:5] /= scaling_factor

                for i in range(output.shape[0]):
                    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
                
                targets = ['person']
                delete = []
                detections = output[:, 1:]

                if detections is not None and len(detections[0]) == 7:
                    for idx, det in enumerate(detections):
                        if classes[int(det[-1])] not in targets:
                            delete.append(idx)
                    detections = np.delete(detections.cpu(), delete, axis=0)
                    tracked_objects = mot_tracker.update_trigger(detections.cpu())
                    # tracked_objects2 = mot_tracker2.update(detections.cpu())
                    # list(map(lambda x: hawkEye.write(x, orig_im), detections))
                    list(map(lambda x: hawkEye.write(x, orig_im), tracked_objects))
                if hawkEye.y_axios > 0:
                    if hawkEye.clickFlag:
                        hawkEye.clickFlag = False
                        mot_tracker.setY_axios(hawkEye.y_axios)
                    cv2.line(orig_im,(0, hawkEye.y_axios), (frame_width, hawkEye.y_axios), (128,25,185), 3)
                    # test.write(detections,tracked_objects,hawkEye.y_axios)
                    # cv2.imwrite("./imgs/frame"+str(frames)+".jpg",orig_im)
                out.write(orig_im)
                # test.write([len(detections), len(tracked_objects), len(tracked_objects2)])
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                # if frames == 60:
                #     break
        else:
            break
    test.endWrite()
    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
