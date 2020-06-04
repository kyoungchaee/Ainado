
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

my_model = [0,0,0]

data_path = 'f_images/front/'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]

    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if images is None:
        continue

    Training_Data.append(np.asarray(images, dtype=np.uint8))

    Labels.append(i)


if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)

my_model[0] = cv2.face.LBPHFaceRecognizer_create()

my_model[0].train(np.asarray(Training_Data), np.asarray(Labels))


#print(my_model.train(np.asarray(Training_Data), np.asarray(Labels)))
print("Front face Model Training Completed!!!!!")


data_path = 'f_images/left/'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]

    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if images is None:
        continue

    Training_Data.append(np.asarray(images, dtype=np.uint8))

    Labels.append(i)


if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)

my_model[1] = cv2.face.LBPHFaceRecognizer_create()

my_model[1].train(np.asarray(Training_Data), np.asarray(Labels))


#print(my_model.train(np.asarray(Training_Data), np.asarray(Labels)))
print("Left face Model Training Completed!!!!!")



data_path = 'f_images/right/'

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]

    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if images is None:
        continue

    Training_Data.append(np.asarray(images, dtype=np.uint8))

    Labels.append(i)


if len(Labels) == 0:
    print("There is no data to train.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)

my_model[2] = cv2.face.LBPHFaceRecognizer_create()

my_model[2].train(np.asarray(Training_Data), np.asarray(Labels))


#print(my_model.train(np.asarray(Training_Data), np.asarray(Labels)))
print("Right face Model Training Completed!!!!!")








#python run_video_face_detect_copy_test.py --net_type slim --test_device cpu --input_size 320

import argparse
import sys
import cv2

import time

import copy

from vision.ssd.config.fd_config import define_img_size


###성능 테스트###
nowtime = time.time()
face_count = 0
total_frame = 0


###############################
#input_img_size = args.input_size
input_img_size = 320
threshold = 0.6
candidate_size = 1000

define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/voc-model-labels.txt"

net_type = 'slim'




videoFile = 0
cap = cv2.VideoCapture(0) #영상 장치 인덱스에 따라 작동

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = cap.get(cv2.CAP_PROP_FPS)



fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('self.mp4', fourcc, 7, (int(width), int(height))) #결과 영상을 self.mp4로 저장
#내꺼 7

################################
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
#test_device = args.test_device
test_device = 'cuda'

#candidate_size = args.candidate_size
#threshold = args.threshold







if net_type == 'slim':
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




###############
##여기 수정함##
###############



def get_circle_x(x1, y1, x2, y2, y):
    a = x2 - x1
    b = y2 - y1
    x = ((x2+x1)/2 + ((x2-x1)/2)*(1-((y-(y2+y1)/2)**2)/((y2-y1)/2)**2)**0.5,(x2+x1)/2 - ((x2-x1)/2)*(1-((y-(y2+y1)/2)**2)/((y2-y1)/2)**2)**0.5 )
#    print(x)
    return x

def blur_circle(img, x1, y1, x2, y2):

    img_blur = copy.deepcopy(img)

    for i in range(int(img_blur.shape[0]/10)):
        for j in range(int(img_blur.shape[1]/10)):
            img_blur[i*10:(i+1)*10,j*10:(j+1)*10] = cv2.blur(img_blur[i*10:(i+1)*10,j*10:(j+1)*10],(50,50))

    yi = y1
    while True:
        if yi == y2:
            break
        img[yi, round(get_circle_x(x1, y1, x2, y2, yi)[1]):round(get_circle_x(x1, y1, x2, y2, yi)[0])] = img_blur[yi, round(get_circle_x(x1, y1, x2, y2, yi)[1]):round(get_circle_x(x1, y1, x2, y2, yi)[0])]
        yi += 1

    return img



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


def nothing(x):
    pass
x_pos_1, y_pos_1 = 0,0
x_pos_2, y_pos_2 = 0,0
unmosaic = False


global x_pos_
global y_pos_
global x_dis_
global y_dis_

x_pos_ = []
y_pos_ = []
x_dis_ = []
y_dis_ = []

i = 0


drawing = False
def mouse_callback(event, x,y, flags, param):

    global i
    global test
    global op

    if event == cv2.EVENT_LBUTTONDOWN: #마우스를 누른 상태, left mouse button being clicked
        drawing = True
        x_pos_.append(x)
        y_pos_.append(y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_pos_.append(x)
        y_pos_.append(y)



    elif event == 8:
        if len(x_pos_)!=0:
            for i in range(len(x_pos_)):
                if min(x_pos_[i],x_pos_[i+1])<= x <= max(x_pos_[i],x_pos_[i+1]):
                    x_pos_[i] = 0
                    y_pos_[i] = 0
                    x_pos_[i+1] = 1
                    y_pos_[i+1] = 1
                    i = i+2






cv2.namedWindow('VideoFrame', cv2.WINDOW_NORMAL)


cv2.setMouseCallback('VideoFrame', mouse_callback)



#dnn part

model = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'deploy.prototxt'


#net, cap open check
if not cap.isOpened():
    print('Camera open failed!')
    exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    exit()



#cv2.namedWindow('VideoFrame',)

while True:

#    cv2.resizeWindow('VideoFrame', (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

#    img_blur = copy.deepcopy(img)
    ret, orig_image = cap.read()
    orig_image_tmp = copy.deepcopy(orig_image)

    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    """

    if orig_image is None:
        print("end")
        break



    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)



    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)

    interval = timer.end()


    for i in range(boxes.size(0)):
        box = boxes[i, :]
     #   label = f" {probs[i]:.2f}"
        if box[1] < 0:
            box[1] = 0
        if box[0] < 0:
            box[0] = 0
        orig_image = blur_circle(orig_image, int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        right_count = 0
        for s in range(3):
            face = cv2.resize(orig_image_tmp[int(box[1]):int(box[3]),int(box[0]):int(box[2])],(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            result = my_model[s].predict(face)
          #  print(result[0]) #닮은 사진, 
            print(result[1]) #유사도 낮을수록 유사한 것.
            if result[1] < 80:
#                confidence_my = int(100*(1-(result[1])/300))
                my_face = orig_image_tmp[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                orig_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = my_face
            if result[1]   > 80:
                right_count += 1
        if right_count > 0:
            orig_image = blur_circle(orig_image, int(box[0]), int(box[1]), int(box[2]), int(box[3]))



#            if confidence_my > 77:
#                print('kim')
#                my_face = orig_image_tmp[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
#                orig_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = my_face

    if len(x_pos_)!=0:
        for i in range(len(x_pos_)-1):
            if i %2 != 0: continue
            if x_pos_[i] == 0 and y_pos_[i] == 0 and x_pos_[i+1] == 1 and y_pos_[i+1] == 1:
                orig_image = blur_circle(orig_image, x_pos_[i], y_pos_[i], x_pos_[i+1], y_pos_[i+1])
            elif y_pos_[i] > y_pos_[i+1] and x_pos_[i] > x_pos_[i+1]:
                orig_image = blur_circle(orig_image, x_pos_[i+1], y_pos_[i+1], x_pos_[i], y_pos_[i])
            elif y_pos_[i]>y_pos_[i+1] and x_pos_[i] < x_pos_[i+1]:
                orig_image = blur_circle(orig_image, x_pos_[i], y_pos_[i+1], x_pos_[i+1], y_pos_[i])
            elif y_pos_[i]<y_pos_[i+1] and x_pos_[i] > x_pos_[i+1]:
                orig_image = blur_circle(orig_image, x_pos_[i+1], y_pos_[i], x_pos_[i], y_pos_[i+1])
            elif y_pos_[i]<y_pos_[i+1] and x_pos_[i] < x_pos_[i+1]:
                orig_image = blur_circle(orig_image, x_pos_[i], y_pos_[i], x_pos_[i+1], y_pos_[i+1])
    #    unmosaic = False

    sum += boxes.size(0)
    cv2.imshow('VideoFrame', orig_image)
    out.write(orig_image)
    if cv2.waitKey(1) > 0:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
