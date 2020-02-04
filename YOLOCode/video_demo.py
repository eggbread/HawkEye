from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse
from sort import *
from PIL import Image

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

# def write(x, img):
#     x = torch.tensor(x)
#     c1 = tuple(x[0:2].int())
#     c2 = tuple(x[2:4].int())
#     cls = int(x[-1])
#     label = "{0}".format(classes[cls])
#     label+=" "+str(int(x[4].item()))+" "
#     confidence = int((x[3]*x[4]).item())
#     if confidence==0:
#       return;
#     label += str(confidence)
#     color = random.choice(colors)
#     cv2.rectangle(img, c1, c2,color, 3)
#     t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
#     c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
#
#     cv2.rectangle(img, c1, c2,color, -1)
#     cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
#     return img
def write(x,  img, tracker_object_data):
    before_x_data = x
    min_threshold = 2
    x = torch.tensor(x)
    c1 = tuple(x[0:2].int())
    c2 = tuple(x[2:4].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    label += " " + str(int(x[4].item())) + " "
    confidence = int(((x[2]-x[0])*(x[3]-x[1])).item())
    if confidence == 0:
        return
    if confidence < 0:
        confidence = -confidence
    label += str(confidence) + " "
    try:
        trackable_object = tracker_objects_data.get(str(int(before_x_data[4])))
        all_y_data = [c[1] for c in trackable_object.centroids]
        mean_result = np.mean(all_y_data)
        direction = mean_result - before_x_data[1]
        if mean_result - min_threshold > direction or direction > mean_result + min_threshold:
            if trackable_object.text == -100:
                label += "ready"
            elif direction < 0 and (trackable_object.text == 1 or tracked_objects.text == 0):
                label += " in"
            else:
                label += " out"
        else:
            label += " stable"
    except:
        label += " stable"
    color = colors[((cls+1) * int(x[4])) % len(colors)]
    cv2.rectangle(img, c1, c2, color, 3)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
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
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
#     model(get_test_input(inp_dim, CUDA), CUDA)

#     model.eval()
    
    videofile = args.video
    cap = cv2.VideoCapture(videofile)
    
    # assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    
    if (cap.isOpened() == False): 
      print("Unable to read camera feed")
    
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    mot_tracker = Sort() 
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
    pre_output = None
    while cap.isOpened():
        
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frames % 3 != 0:
            frames += 1
            continue
        if ret:
            
            
            img, orig_im, dim = prep_image(frame, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            
            detections = output[:,1:]
            if detections is not None and len(detections[0]) == 7:
                acc = 0
                while True:
                    mot_tracker.update(detections.cpu())
                    tracked_objects, tracker_objects_data = mot_tracker.update(detections.cpu())
                    acc = len(tracked_objects) / len(detections)
                    if frames == 0:
                        break
                    if acc > 0.7:
                        list(map(lambda x: write(x, orig_im, tracker_objects_data), tracked_objects))
                        break
                print("Accuracy : ",acc*100,"%")
            # if detections is not None:
            #     tracked_objects = mot_tracker.update(detections.cpu())
            #     list(map(lambda x: write(x, orig_im), tracked_objects))
            #     print("Accuracy : ", len(tracked_objects) / len(detections) * 100, "%")

                # pre_output = tracked_objects
            out.write(orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            if frames == 200:
                break
            
        else:
            break
    
    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()
    
    # Closes all the frames
    cv2.destroyAllWindows() 

    
    

