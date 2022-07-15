def model1():
    model_1 = Sequential()
    model_1.add(
        Convolution2D(filters=1, kernel_size=kernel_size1, strides=(1, 1), padding='VALID', input_shape=(77, 3, 1),
                      kernel_initializer='random_uniform', bias_initializer='zeros'))  # 卷积层1
    model_1.add(LeakyReLU(alpha=0.05))  # 激活层
    model_1.add(MaxPooling2D(pool_size=pool_size1))  # 池化层
    return model_1


def model2():
    model_2 = Sequential()
    model_2.add(Convolution2D(filters=1, kernel_size=kernel_size2, strides=(1, 1), padding='VALID',
                              input_shape=(77, 3, 1)))  # 卷积层1
    model_2.add(LeakyReLU(alpha=0.05))  # 激活层
    model_2.add(MaxPooling2D(pool_size=pool_size2))  # 池化层
    return model_2


def model3():
    model_3 = Sequential()
    model_3.add(Convolution2D(filters=1, kernel_size=kernel_size3, strides=(1, 1), padding='VALID',
                              input_shape=(77, 3, 1)))  # 卷积层1
    model_3.add(LeakyReLU(alpha=0.05))  # 激活层
    model_3.add(MaxPooling2D(pool_size=pool_size3))  # 池化层
    return model_3


def merge_model():
    model_1 = model1()
    model_2 = model2()
    model_3 = model3()

    inp1 = model_1.input  # 参数在这里定义
    inp2 = model_2.input  # 第二个模型的参数
    inp3 = model_3.input
    r1 = model_1.output
    r2 = model_2.output
    r3 = model_3.output
    x = keras.layers.Concatenate(axis=1)([r1, r2, r3])
    model = Model(inputs=[inp1, inp2, inp3], outputs=x)
    return model

merged_model = merge_model()

inp = merged_model.input
x = merged_model.output
print(x.shape)

# def slice(x,index):
#     return x[:,:,:,index]
# x=Lambda(slice,arguments={'index':0})(x)
# x.squeeze()
# print(x)
# x = Reshape((57))(x)
x = Flatten()(x)
print(x.shape)
# lstm1 = LSTM(128,activation='relu',return_sequences=True)(x)
# lstm1 = Dropout(0.2)(lstm1)
# lstm2 = LSTM(256,return_sequences=False)(lstm1)
# lstm2 = Dropout(0.2)(lstm2)

den = Dense(128)(x)
l = LeakyReLU(alpha = 0.05)(den) #激活层
l = Dropout(0.2)(l)
# l = Flatten()(l)

den = Dense(32)(l)
l = LeakyReLU(alpha = 0.05)(den) #激活层
l = Dropout(0.2)(l)
# l = Flatten()(l)

den = Dense(8)(l)
l = LeakyReLU(alpha = 0.05)(den) #激活层
l = Dropout(0.2)(l)
# l = Flatten()(l)

# attention_probs = Dense(7296,activation='softmax', name='attention_probs')(l)
# attention_mul = merge([l, attention_probs],output_shape=32, name='attention_mul', mode='mul')
# attention_mul = Dense(64)(attention_mul)

result = Dense(1,activation='sigmoid')(l)

model = Model(inputs = inp,outputs=result)