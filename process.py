from skimage.feature import hog
import cv2 as cv
import numpy as np

import joblib

model_best = 'model/drone_bird_modelv2.sav'
model_old = 'model/drone_bird_modelv1.sav'

model = joblib.load(model_best)
model_nohog = joblib.load(model_old)

def preprocess_image(img, target_size=(128, 72), hog_=True):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
    img_blur = cv.bilateralFilter(img, 7, 75, 100)

    laplacian = cv.Laplacian(img_blur, cv.CV_64F, ksize=3)
    img_laplacian = cv.convertScaleAbs(img_blur - laplacian)
    image_data = img_laplacian

    if hog_:
        hog_features, hog_image = hog(img_laplacian, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                                  block_norm='L2-Hys', visualize=True, transform_sqrt=True)
        image_data = hog_features
    return image_data


def predict_result(predict_img, hog_=True):
    print(model)
    if hog_:
        result = model.predict(predict_img.reshape(1, -1))
    else:
        result = model_nohog.predict(predict_img.reshape(1, -1))
    # confidence = model.predict_proba(predict_img.reshape(1, -1))
    return result
