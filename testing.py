import numpy as np
import cv2
import os
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model


img=cv2.imread(r'F:\python\Deep learning\cat_dog_classification\cat.3.jfif')
cv2.imshow('img',img)
model = tf.keras.models.load_model('cat_dog.h5')
test_img=img.resize(250,250,1)
test_img=image.img_to_array(img)
test_img=np.expand_dims(test_img,axis=0)
result=model.predict_classes(test_img)
print(result)
