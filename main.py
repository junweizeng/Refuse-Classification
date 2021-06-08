# encoding:utf-8
'''
调用各个模块实现对图片的测试和显示结果
'''
from plot import show
import predict
from prepare_image import prepare_image
from flask import *
from flask_cors import *


def detect(img_path):
    model = predict.model()
    results = prepare_image(img_path=img_path, model=model)
    return show(img_path=img_path, results=results)


app = Flask(__name__)
# 跨域
CORS(app, supports_credentials=True)


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    print(request.files)
    img = request.files.get('file')
    file_path = 'catch/' + img.filename
    img.save(file_path)
    return detect(file_path)


# if __name__ == '__main__':
#     detect("test4.jpg")
app.run()
