# encoding:utf-8
"""
用于训练垃圾分类模型
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import ModelCheckpoint
from keras.applications.densenet import DenseNet201
from keras import backend as K

# 图片长和宽的值
img_width, img_height = 512, 384

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2527
nb_validation_samples = 189
epochs = 70
batch_size = 20

# 判断数据的通道维度的位置
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# 初始化模型，用来添加下面几层
model = Sequential()
# 添加DenseNet201模型，使用单卷积层和最大池化层，建立的是前面所有层与后面层的密集连接，优化训练出来的模型
denseNet = DenseNet201(input_shape=input_shape, weights='imagenet', pooling='avg', include_top=False)
# 设置为不可训练的
denseNet.trainable = False
model.add(denseNet)
# 添加全连接层，起到分类的作用
model.add(Dense(128))
# 添加激活函数
model.add(Activation('relu'))
# 添加全连接层，输出6种结果
model.add(Dense(6))
# 采用Softmax，用于多分类神经网络输出
model.add(Activation('softmax'))

# 编译模型，为模型设定训练时采用的优化函数、损失函数
model.compile(loss='categorical_crossentropy',  # 多分类
              optimizer='rmsprop',
              metrics=['accuracy'])

# 训练集，图片生成器，对图片进行数据增强处理（翻转等）
# rescale：重缩放因子；对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，在一些模型当中，直接输入原图的像素值可能会落入激活函数的“死亡区”，因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”
# shear_range：错切变换；x轴或y轴随机平移的范围
# zoom_range：随机缩放范围；可以让图片在长或宽的方向进行放大，而参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作。
# horizontal_flip：随机水平翻转
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# 测试集，图片生成器
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 使用图片生成器对训练集进行相应的图像操作，并返回操作完成后的图像数据集
train_generator = train_datagen.flow_from_directory(
    train_data_dir,  # 训练集的路径
    target_size=(img_width, img_height),  # 目标图像大小
    batch_size=batch_size,  # 批量数据代销，这里设置为20
    class_mode='categorical')  # 多分类

# 使用图片生成器对测试集进行相应的图像操作，并返回操作完成后的图像数据集
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,  # 测试集的路径
    target_size=(img_width, img_height),  # 目标图像大小
    batch_size=batch_size,  # 批量数据代销，这里设置为20
    class_mode='categorical')  # 多分类

filepath = "weights5.h5"
# 保存训练结果
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False,
                             mode='auto', period=1)
callbacks_list = [checkpoint]

# 训练模型
model.fit_generator(
    # 训练数据集
    train_generator,
    # 每步训练张数
    steps_per_epoch=nb_train_samples // batch_size,
    # 训练轮数
    epochs=epochs,
    # 回调函数，用来最后保存训练模型
    callbacks=callbacks_list,
    # 测试数据集
    validation_data=validation_generator,
    # 步长
    validation_steps=nb_validation_samples // batch_size)
