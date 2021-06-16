# encoding:utf-8
"""
调用各个模块实现对图片的测试和结果返回
"""

from prepare_image import prepare_image
from flask import *
from flask_cors import *
from result import *
from keras import *


model = models.load_model('./data/weight/weights3.h5')


def detect(image_path):
    # 返回图片的预测结果
    result = prepare_image(image_path=image_path, model=model)
    # 处理预测结果，返回垃圾类型
    garbage = generate_result(result=result)
    # 对结果进行分类
    if garbage == "trash":
        garbage = "干垃圾 干垃圾"
    elif garbage == "paper":
        garbage = "纸张 可回收垃圾"
    elif garbage == "cardboard":
        garbage = "硬纸板 可回收垃圾"
    elif garbage == "glass":
        garbage = "玻璃垃圾 可回收垃圾"
    elif garbage == "metal":
        garbage = "金属垃圾 可回收垃圾"
    elif garbage == "plastic":
        garbage = "塑料垃圾 可回收垃圾"
    return garbage


app = Flask(__name__)
# 跨域
CORS(app, supports_credentials=True)


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    print(request.files)
    im = request.files.get('file')
    file_path = 'catch/' + im.filename
    im.save(file_path)
    return detect(file_path)


app.run()
