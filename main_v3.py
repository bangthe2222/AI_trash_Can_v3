from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QImage, QPixmap 
import sys
import serial
import keyboard
import time
import cv2
import numpy as np
import sqlite3
import random
import tensorflow.lite as tflite

def findData(cursor,id):
    cursor.execute("SELECT * FROM DATA WHERE PHONE=?", (id,))
    rows = cursor.fetchall()
    # print("select")
    return rows

def updateData(cursor,id):
    try:
        cursor.execute('''INSERT INTO DATA VALUES (?, ?)''',(id,0))
        print("create")
        conn.commit()
    except:    
        data = findData(cursor,id)
        num = data[0][1]+1
        cursor.execute('''UPDATE DATA SET NUM = ? WHERE PHONE=?;''',(num,id))
        print("update")
        conn.commit()

def getGift(cursor,id):
    data = findData(cursor,id)
    num = data[0][1]-5
    cursor.execute('''UPDATE DATA SET NUM = ? WHERE PHONE=?;''',(num,id))
    print("update")
    conn.commit()

def getAllData(cursor):
    print("Data Inserted in the table: ")
    data=cursor.execute('''SELECT * FROM DATA''')
    for row in data:
        print(row)   
    return data
def letterbox(im, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)
def detect(img):
    # img = cv2.imread('shape.jpg')
    check_bottle = False
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray([img], dtype= np.float32)
    # im /= 255
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    id_out = np.argmax(output_data[0])

    # get accuracy
    acc_pre = output_data[0][id_out]

    if id_out == 3 and acc_pre > 0.8:
        check_bottle = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # writer.write(image)
    if check_bottle:
        return image, True
    return image, False

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        # root config
        uic.loadUi('screen.ui', self)
        self.setWindowTitle("MÁY AI THU CHAI NHỰA ĐỔI QUÀ")
        # const 
        self.cap = cv2.VideoCapture(0)
        self.check_bottle = False
        self.user_id = ""
        self.num_bottle = 0
        self.total_bottle = 0
        self.warning = "Xin chào quý khách"
        self.check_user = False

        # TIMER 
        self.timer_root = QtCore.QTimer(self)
        self.timer_root.timeout.connect(self.show_detect)
        # self.timer_root.timeout.connect(self.checkBottle)

        self.timer_1 = QtCore.QTimer(self)
        self.timer_1.timeout.connect(self.getId)

        self.timer_1_sec = QtCore.QTimer(self)
        self.timer_1_sec.timeout.connect(self.checkOut)
        self.wait_time = 0

        self.timer_root.start(1)
        self.timer_1.start(10)

        self.timer_win = QtCore.QTimer(self)
        self.timer_win.timeout.connect(self.showNumBot)
        self.timer_win.start(10)
        self.show()
    def show_detect(self):
        _, self.image = self.cap.read()
        self.image, self.check_bottle = detect(self.image)
        self.image = cv2.resize(self.image, (480,320))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.convert = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format.Format_RGB888)
        self.frame.setPixmap(QPixmap.fromImage(self.convert))
    def show_frame(self):
        _, self.image = self.cap.read()
        self.image = cv2.resize(self.image, (480,320))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.convert = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format.Format_RGB888)
        self.frame.setPixmap(QPixmap.fromImage(self.convert))        
    def checkBottle(self):
        x = ser.readline()
        print(x)
        # print(result)
        if (x[:-2].decode("utf-8")) == "0001":
            time.sleep(0.1)
            self.show_detect()
            # result = detect(frame, net)
            if self.check_bottle :
                ser.write("0001".encode("utf-8"))
                print("bottle here")
                time.sleep(0.01)
                self.num_bottle+=1
                if self.check_user == True:
                    updateData(cursor, self.user_id)
                    self.timer_1_sec.start(1000)
            else:
                ser.write("0000".encode("utf-8"))
                time.sleep(0.01)
    def getId(self):
        list_num = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        self.user_id =  self.phoneid.text()
        if keyboard.is_pressed("enter"):
            if 10<=len(self.user_id)<12:
                for i in self.user_id:
                    if i not in list_num:
                        self.warning = "Mời nhập lại số điện thoại"
                        self.phoneid.clear()
                        self.user_id = ""
                        
                if len(self.user_id) >0:
                    self.warning = "Đăng nhập thành công"
                    self.check_user = True
            else:
                self.warning = "Mời nhập số điện thoại"
                self.phoneid.clear()
                self.user_id = ""
        elif keyboard.is_pressed("*"):
                # self.warning = "Hẹn gặp lại"
                self.reset()
        elif keyboard.is_pressed("tab"):
            print("pass")
            print(self.total_bottle)
            print(self.check_user)
            if self.total_bottle >=5 and self.check_user:
                print("hello")
                getGift(cursor, self.user_id)
                # ser.write("0011".encode("utf-8"))
                time.sleep(1)
            else:
                self.warning = "Phải có ít nhất 5 chai"
            
    def checkOut(self):
        self.wait_time += 1
        if self.wait_time == 10:
            self.reset()
            self.timer_1_sec.stop()

    def showNumBot(self):
        self.total.setText(str(self.total_bottle))
        self.numbot.setText(str(self.num_bottle))
        self.warning_label.setText(self.warning)
        if self.check_user:
            data_bot = findData(cursor,self.user_id)
            if data_bot:
                self.total.setText(str(data_bot[0][1]))
                self.total_bottle = data_bot[0][1]
            else:
                self.total.setText("0")
    
    def reset(self):
        self.phoneid.clear()
        self.check_bottle = False
        self.user_id = ""
        self.num_bottle = 0
        self.total_bottle = 0
        self.warning = "Xin chào quý khách"
        self.check_user = False
if __name__ == "__main__":
    ser = serial.Serial("COM6",9600, timeout = 0.1)
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")
    cursor = conn.cursor()
    # load our serialized model from disk
    print("[INFO] loading model...")

        #Name of the classes according to class indices.
    names = ['Alu', 'Foam_box', 'Milk_box', 'PET', 'Paper', 'Paper_cup', 'Plastic_cup']

    #Creating random colors for bounding box visualization.
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="./InceptionV3_7class_60_epoch.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    cap = cv2.VideoCapture(0)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] starting video stream...")
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec())
    