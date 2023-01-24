import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow import keras
from skimage.transform import resize, pyramid_reduce
import PIL
from PIL import Image
import matplotlib.pyplot as plt

model1 = load_model('model1')
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def prediction(pred):
    return(chr(pred+ 65))


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )
    
    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (1,28,28), interpolation = cv2.INTER_AREA)
  
    return img
 

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

def main():
    l = []
    
    while True:
        
        cv2.waitKey(20)
        cam_capture = cv2.VideoCapture(0)
        print("succsess captured")
        _, image_frame = cam_capture.read()  
    # Select ROI(The Blue Box)
        im2 = crop_image(image_frame, 300,300,300,300)
        print('passed')
        image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
        #image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
        im3 = cv2.resize(image_grayscale, (28,28), interpolation = cv2.INTER_AREA)


    
        im4 = np.resize(im3, (28, 28, 1))
        im5 = np.expand_dims(im4, axis=0)
    

        pred_probab, pred_class = keras_predict(model1, im5)
    
        curr = prediction(pred_class)
        print(curr)
        cv2.putText(image_frame, curr, (200, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
            
            
    
 
    # Display cropped image
        cv2.rectangle(image_frame, (300, 300), (600, 600), (255, 255, 00), 3)
        cv2.imshow("frame",image_frame)
        
        
        
   
        
        cv2.imshow("Image3",image_grayscale)
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    
    
    cam_capture.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main()
   