import sys
import cv2
import time
import torch
import os
import numpy as np
import copy
from vision.ssd.config.fd_config import define_img_size
global i

i = 0
x_pos_ = []
y_pos_ = []
x_dis_ = []#x점 두 개 사이의 거리
y_dis_ = []#y점 두 개 사이의 거리


###성능 테스트###
nowtime = time.time()
face_count = 0
total_frame = 0


###############################
#input_img_size = args.input_size
input_img_size = 320
threshold = 0.7
candidate_size = 1000

define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    if event == 1:
        global x_pos_1
        global y_pos_1
        x_pos_1 = x
        y_pos_1 = y
    elif event == 4:
        global x_pos_2
        global y_pos_2
        global x_dis
        global y_dis
        x_pos_2 = x
        y_pos_2 = y
        x_dis = x_pos_2 - x_pos_1
        y_dis = y_pos_2 - y_pos_1
#        frame_tmp2[int(capture.get(cv2.CAP_PROP_POS_FRAMES))] = [x_pos_1, y_pos_1, x_pos_2, y_pos_2]
 #       print(frame_tmp2)
    elif flags == 8 and event == 0:
        x_pos_1 = x - int(x_dis/2)
        y_pos_1 = y - int(y_dis/2)
        x_pos_2 = x + int(x_dis/2)
        y_pos_2 = y + int(y_dis/2)
        frame_tmp2[int(cap.get(cv2.CAP_PROP_POS_FRAMES))] = [x_pos_1, y_pos_1, x_pos_2, y_pos_2]
    elif flags == 32 and event == 0:
        if frame_tmp2.get(int(cap.get(cv2.CAP_PROP_POS_FRAMES))) != None:
            global mosaic
            mosaic = 1

def get_circle_x(x1, y1, x2, y2, y):
    a = x2 - x1
    b = y2 - y1
    x = ((x2 + x1) / 2 + ((x2 - x1) / 2) * (1 - ((y - (y2 + y1) / 2) ** 2) / ((y2 - y1) / 2) ** 2) ** 0.5,
         (x2 + x1) / 2 - ((x2 - x1) / 2) * (1 - ((y - (y2 + y1) / 2) ** 2) / ((y2 - y1) / 2) ** 2) ** 0.5)
    #    print(x)
    return x

def blur_circle(img, x1, y1, x2, y2): #모자이크 영역 설정
    #   print(img[10, x1:int(get_circle_x(x1, y1, x2, y2, 10)[1])])

    img_blur = copy.deepcopy(img)
    #    img_blur = img[:]
    img_blur[y1:y2, x1:x2] = cv2.blur(img_blur[y1:y2, x1:x2], (100, 100))

    yi = y1
    while True:
        if yi == y2:
            break
        img[yi, round(get_circle_x(x1, y1, x2, y2, yi)[1]):round(get_circle_x(x1, y1, x2, y2, yi)[0])] = img_blur[yi,
                                                                                                         round(
                                                                                                             get_circle_x(
                                                                                                                 x1, y1,
                                                                                                                 x2, y2,
                                                                                                                 yi)[
                                                                                                                 1]):round(
                                                                                                             get_circle_x(
                                                                                                                 x1, y1,
                                                                                                                 x2, y2,
                                                                                                                 yi)[
                                                                                                                 0])]
        #        img[yi, 30:60] = img_blur[yi, 30:60]
        yi += 1

    return img

def draw_circle(event, x, y, flags, param): #마우스 클릭 이벤트를 통한 모자이크 처리
    print(i)

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        x_pos_.append(x)
        y_pos_.append(y)


    elif event == cv2.EVENT_LBUTTONUP:
        x_pos_.append(x)
        y_pos_.append(y)

        #         drawing = False
        #         global x_pos_2
        #         global y_pos_2
        #         global x_dis
        #         global y_dis
        #         x_pos_2 = x
        #         y_pos_2 = y

        if x_pos_[i] > x_pos_[i + 1] and y_pos_[i] > y_pos_[i + 1]:
            #             x_dis_[i] = x_pos_[i] - x_pos_[i+1]
            #             y_dis_[i] = y_pos_[i] - y_pos_[i+1]
            x_dis_.append(x_pos_[i] - x_pos_[i + 1])
            y_dis_.append(y_pos_[i] - y_pos_[i + 1])
        elif x_pos_[i] > x_pos_[i + 1] and y_pos_[i] < y_pos_[i + 1]:
            #             x_dis_[i] = x_pos_[i] - x_pos_[i+1]
            #             y_dis_[i] = y_pos_[i+1] - y_pos_[i]
            x_dis_.append(x_pos_[i] - x_pos_[i + 1])
            y_dis_.append(y_pos_[i + 1] - y_pos_[i])
        elif x_pos_[i] < x_pos_[i + 1] and y_pos_[i] > y_pos_[i + 1]:
            #             x_dis_[i] = x_pos_[i+1] - x_pos_[i]
            #             y_dis_[i] = y_pos_[i] - y_pos_[i+1]
            x_dis_.append(x_pos_[i + 1] - x_pos_[i])
            y_dis_.append(y_pos_[i] - y_pos_[i + 1])
        elif x_pos_[i] < x_pos_[i + 1] and y_pos_[i] < y_pos_[i + 1]:
            #             x_dis_[i] = x_pos_[i+1] - x_pos_[i]
            #             y_dis_[i] = y_pos_[i+1] - y_pos_[i]
            x_dis_.append(x_pos_[i + 1] - x_pos_[i])
            y_dis_.append(y_pos_[i + 1] - y_pos_[i])




mouse_event_types = { 0:"EVENT_MOUSEMOVE", 1:"EVENT_LBUTTONDOWN", 2:"EVENT_RBUTTONDOWN", 3:"EVENT_MBUTTONDOWN",
                 4:"EVENT_LBUTTONUP", 5:"EVENT_RBUTTONUP", 6:"EVENT_MBUTTONUP",
                 7:"EVENT_LBUTTONDBLCLK", 8:"EVENT_RBUTTONDBLCLK", 9:"EVENT_MBUTTONDBLCLK",
                 10:"EVENT_MOUSEWHEEL", 11:"EVENT_MOUSEHWHEEL"}
mouse_event_flags = { 0:"None", 1:"EVENT_FLAG_LBUTTON", 2:"EVENT_FLAG_RBUTTON", 4:"EVENT_FLAG_MBUTTON",
                8:"EVENT_FLAG_CTRLKEY", 9:"EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON",
                10:"EVENT_FLAG_CTRLKEY + EVENT_FLAG_RBUTTON", 11:"EVENT_FLAG_CTRLKEY + EVENT_FLAG_MBUTTON",
                16:"EVENT_FLAG_SHIFTKEY", 17:"EVENT_FLAG_SHIFTKEY + EVENT_FLAG_LBUTTON",
                18:"EVENT_FLAG_SHIFTLKEY + EVENT_FLAG_RBUTTON", 19:"EVENT_FLAG_SHIFTKEY + EVENT_FLAG_MBUTTON",
                32:"EVENT_FLAG_ALTKEY", 33:"EVENT_FLAG_ALTKEY + EVENT_FLAG_LBUTTON",
                34:"EVENT_FLAG_ALTKEY + EVENT_FLAG_RBUTTON", 35:"EVENT_FLAG_ALTKEY + EVENT_FLAG_MBUTTON"}

label_path = "./models/voc-model-labels.txt"

#####################
#net_type = args.net_type
net_type = 'slim'


##############경로 수정함###########
#cap = cv2.VideoCapture(args.video_path)  # capture from video
#cap = cv2.VideoCapture(0)  # capture from camera / webcam
file_name = 'travel.mp4'
cap = cv2.VideoCapture(file_name) # capture from video

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX') #if you want to use DIVX codec to decode .avi
out = cv2.VideoWriter('SAVE_travel.avi', fourcc, fps, (int(width), int(height)))

#fourcc = cv2.VideoWriter_fourcc(*'MPEG')
#out = cv2.VideoWriter('SAVE_travel.mp4', fourcc, fps, (int(width), int(height)))

################################
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
#test_device = args.test_device

test_device = 'cuda'

#candidate_size = args.candidate_size
#threshold = args.threshold

frame_tmp = []
frame_tmp2 = {}
proc = 0 #현재 진행 상황
###전체 불러오기

x_pos_1, y_pos_1, x_pos_2, y_pos_2 = 0 ,0 ,1,1

mosaic = False
video_speed = 10 #1 이상의 정수

if net_type == 'slim': #ultra fast light 모드 중 하나, slim이 더 빠름
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)

elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)

else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)


timer = Timer()
sum = 0
cap = cv2.VideoCapture(file_name) # capture from video
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



while True:
    ret, orig_image= cap.read()
    frame_tmp.append(orig_image)
    orig_image_tmp = copy.deepcopy(orig_image)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == 300:
       break
    if not ret:
        break
    proc += 1

cap.release()

cap = cv2.VideoCapture(file_name) # capture from video
cv2.namedWindow(file_name)
cv2.createTrackbar('Frame', file_name, 0, total_frame, nothing)
cv2.createTrackbar("Pause", file_name, 0, 1, nothing)

while True:

    ret, orig_image = cap.read()
    cv2.setMouseCallback(file_name, mouse_callback)
    cv2.setMouseCallback(file_name, draw_circle)
    p = cv2.getTrackbarPos('Pause', file_name)
    r = cv2.getTrackbarPos('Frame', file_name)

    timer.start()
    image =cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    r = r + 1
    cv2.setTrackbarPos('Frame', file_name, r)

    #cap.set(cv2.CAP_PROP_POS_FRAMES, r)
    #cv2.setTrackbarPos('Pause', file_name, p)
    #cv2.getTrackbarPos('Frame', file_name)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        #   label = f" {probs[i]:.2f}"
        if box[1] < 0:
            box[1] = 0
        if box[0] < 0:
            box[0] = 0

        #        blur_img = cv2.blur(orig_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])], (100, 100))
        #        orig_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_circle(orig_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])], int(box[1]), int(box[3]), int(box[0]), int(box[2]))
        orig_image = blur_circle(orig_image, int(box[0]), int(box[1]), int(box[2]), int(box[3]))

    sum += boxes.size(0)

    if len(x_pos_) != 0:
        for i in range(len(x_pos_) - 1):
            if i % 2 != 0: continue
            if y_pos_[i] > y_pos_[i + 1] and x_pos_[i] > x_pos_[i + 1]:
                orig_image[y_pos_[i + 1]:y_pos_[i], x_pos_[i + 1]:x_pos_[i]] = cv2.blur(
                    orig_image[y_pos_[i + 1]:y_pos_[i], x_pos_[i + 1]:x_pos_[i]], (100, 100))
            elif y_pos_[i] > y_pos_[i + 1] and x_pos_[i] < x_pos_[i + 1]:
                orig_image[y_pos_[i + 1]:y_pos_[i], x_pos_[i]:x_pos_[i + 1]] = cv2.blur(
                    orig_image[y_pos_[i + 1]:y_pos_[i], x_pos_[i]:x_pos_[i + 1]], (100, 100))
            elif y_pos_[i] < y_pos_[i + 1] and x_pos_[i] > x_pos_[i + 1]:
                orig_image[y_pos_[i]:y_pos_[i + 1], x_pos_[i + 1]:x_pos_[i]] = cv2.blur(
                    orig_image[y_pos_[i]:y_pos_[i + 1], x_pos_[i + 1]:x_pos_[i]], (100, 100))
            elif y_pos_[i] < y_pos_[i + 1] and x_pos_[i] < x_pos_[i + 1]:
                orig_image[y_pos_[i]:y_pos_[i + 1], x_pos_[i]:x_pos_[i + 1]] = cv2.blur(
                    orig_image[y_pos_[i]:y_pos_[i + 1], x_pos_[i]:x_pos_[i + 1]], (100, 100))
        i = i + 2

    #frame = frame_tmp[r]

    out.write(orig_image)


    #    print(cap.get(cv2.CAP_PROP_POS_AVI_RATIO))

    if p == 0:
        cv2.waitKey(-1)

    if cv2.waitKey(1) > 0:

        break

    cap.set(cv2.CAP_PROP_POS_FRAMES, r)
    cv2.imshow(file_name, orig_image)

cap.release()
out.release()
cv2.destroyAllWindows()

