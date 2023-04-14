import tensorflow.lite as tflite
import cv2
import numpy as np
import random
import time
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

#Name of the classes according to class indices.
names =  ['Alu', 'Foam_box', 'Milk_box', 'PET', 'Paper', 'Paper_cup', 'Plastic_cup']

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

# writer= cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))
_ = True
while True:
    if _:
        t1 = time.time()
        _,img = cap.read()
        img_src = img
        # img = cv2.imread('shape.jpg')
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

        print("id: ",names[id_out],"acc: ", acc_pre)
    
        # writer.write(image)
        cv2.imshow("image", img_src)
        print("fps:",1/(time.time()-t1))
        cv2.waitKey(1)
    else:
        break
# writer.release()