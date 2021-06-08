# encoding:utf-8
'''
用于训练垃圾分类模型
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import backend as K

# dimensions of our images.
img_width, img_height = 512, 384

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2527
nb_validation_samples = 189
epochs = 50
batch_size = 20

# 判断数据的通道维度的位置
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
# 添加卷积层，卷积尺寸3*3
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# 添加激活函数
model.add(Activation('relu'))
# 添加最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))  # 6分类
model.add(Activation('softmax'))  # 采用Softmax

# 编译模型，为模型设定训练时采用的优化函数、损失函数
model.compile(loss='categorical_crossentropy',  # 多分类
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # 多分类

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # 多分类

filepath="data/weight/weights2.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

# 训练模型,
model.fit_generator(
    train_generator,
    # 每步训练张数
    steps_per_epoch=nb_train_samples // batch_size,
    # 训练轮数
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

