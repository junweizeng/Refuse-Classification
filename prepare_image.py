# encoding:utf-8
"""
对需要进行预测的图片进行预处理
"""
from classify import *


def prepare_image(image_path, model):
    im = load_img(image_path, target_size=(100, 100))
    im = img_to_array(im)
    im = im / 255.0
    prediction_image = np.array(im)
    prediction_image = np.expand_dims(prediction_image, axis=0)
    results = model.predict(prediction_image)
    print(results)
    return results
