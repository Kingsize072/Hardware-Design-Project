from maix import nn, camera, image, display
from maix.nn.app.face import FaceRecognize
import time
from evdev import InputDevice
from select import select

score_threshold = 70  #70                           
input_size = (224, 224, 3)                      
input_size_fe = (128, 128, 3)                   
feature_len = 256                               
steps = [8, 16, 32]                            
channel_num = 0                                 
users = []                                      
names = ["Not wear Mask!", "Wear Mask!"]
model = {
    "param": "/home/model/face_recognize/model_int8.param",
    "bin": "/home/model/face_recognize/model_int8.bin"
}
model_fe = {
    "param": "/home/model/face_recognize/fe_res18_117.param",
    "bin": "/home/model/face_recognize/fe_res18_117.bin"
}

for i in range(len(steps)):
    channel_num += input_size[1] / steps[i] * (input_size[0] / steps[i]) * 2
channel_num = int(channel_num)     
options = {                             
    "model_type":  "awnn",
    "inputs": {
        "input0": input_size
    },
    "outputs": {
        "output0": (1, 4, channel_num) ,
        "431": (1, 2, channel_num) ,
        "output2": (1, 10, channel_num)
    },
    "mean": [127.5, 127.5, 127.5],
    "norm": [0.0078125, 0.0078125, 0.0078125],
}
options_fe = {                            
    "model_type":  "awnn",
    "inputs": {
        "inputs_blob": input_size_fe
    },
    "outputs": {
        "FC_blob": (1, 1, feature_len)
    },
    "mean": [127.5, 127.5, 127.5],
    "norm": [0.0078125, 0.0078125, 0.0078125],
}
keys = InputDevice('/dev/input/event0')

threshold = 0.5                                        
nms = 0.25 # 3
max_face_num = 1                                       
m = nn.load(model, opt=options)
m_fe = nn.load(model_fe, opt=options_fe)
face_recognizer = FaceRecognize(m, m_fe, feature_len, input_size, threshold, nms, max_face_num)

def get_key():                                      
    r,w,x = select([keys], [], [],0)
    if r:
        for event in keys.read():
            if event.value == 1 and event.code == 0x02:     
                return 1
            elif event.value == 1 and event.code == 0x03:   
                return 2
            elif event.value == 2 and event.code == 0x03:   
                return 3
    return 0

def map_face(box,points):                           
    if display.width() == display.height():
        def tran(x):
            return int(x/224*display.width())
        box = list(map(tran, box))
        def tran_p(p):
            return list(map(tran, p))
        points = list(map(tran_p, points))
    else:
        s = (224*display.height()/display.width())
        w, h, c = display.width()/224, display.height()/224, 224/s
        t, d = c*h, (224 - s) // 2
        box[0], box[1], box[2], box[3] = int(box[0]*w), int((box[1]-28)*t), int(box[2]*w), int((box[3])*t)
        def tran_p(p):
            return [int(p[0]*w), int((p[1]-d)*t)]
        points = list(map(tran_p, points))
    return box,points

def darw_info(draw, box, points, disp_str, bg_color=(255, 0, 0), font_color=(255, 255, 255)):
    box,points = map_face(box,points)
    font_wh = image.get_string_size(disp_str)
    for p in points:
        draw.draw_rectangle(p[0] - 1, p[1] -1, p[0] + 1, p[1] + 1, color=bg_color)
    draw.draw_rectangle(box[0], box[1], box[0] + box[2], box[1] + box[3], color=bg_color, thickness=2)
    draw.draw_rectangle(box[0], box[1] - font_wh[1], box[0] + font_wh[0], box[1], color=bg_color, thickness = -1)
    draw.draw_string(box[0], box[1] - font_wh[1], disp_str, color=font_color)
def recognize(feature):                                                                   
    def _compare(user):                                                         
        return face_recognizer.compare(user, feature)                   
    face_score_l = list(map(_compare,users))                               
    return max(enumerate(face_score_l), key=lambda x: x[-1])                

def run():
    img = camera.capture()                      
    AI_img = img.copy().resize(224, 224)
    if not img:
        time.sleep(0.02)# 0.02
        return
    faces = face_recognizer.get_faces(AI_img.tobytes(),False)
    if faces:
        for prob, box, landmarks, feature in faces:
            key_val = get_key()
            if key_val == 1:                                
                if len(users) < len(names):
                    print("Adding Rec:", len(users))
                    users.append(feature)
                else:
                    print("Rec Full!")
            elif key_val == 2:                              
                if len(users) > 0:
                    print("Remove Rec!:", names[len(users) - 1])
                    users.pop()
                else:
                    print("No Rec!")

            if len(users):                             
                maxIndex = recognize(feature)

                if maxIndex[1] > score_threshold:                                      
                    darw_info(img, box, landmarks, "{}:{:.2f}".format(names[maxIndex[0]], maxIndex[1]), font_color=(0, 0, 255, 255), bg_color=(0, 255, 0, 50))
                    print("{}".format(names[maxIndex[0]], maxIndex[1]))
                else:
                    darw_info(img, box, landmarks, "{}:{:.2f}".format(names[maxIndex[0]], maxIndex[1]), font_color=(255, 255, 255, 255), bg_color=(255, 0, 0, 50))
                    print("May be {}".format(names[maxIndex[0]], maxIndex[1]))
            else:                                           
                darw_info(img, box, landmarks, "Error No Rec!", font_color=(255, 255, 255, 255), bg_color=(255, 0, 0, 50))
    display.show(img)

if __name__ == "__main__":
    import signal
    def handle_signal_z(signum,frame):
        print("Error")
        exit(0)
    signal.signal(signal.SIGINT,handle_signal_z)
    while True:
        run()
