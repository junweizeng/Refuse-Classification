# encoding:utf-8
"""
用于对预测结果图片进行显示
"""

# 根据预测结果显示对应的文字
classes_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def generate_result(result):
    maxNum = 0
    index = 0
    for i in range(6):
        if result[0][i] > maxNum:
            maxNum = result[0][i]
            index = i
    return classes_types[index]

