import keras
import  numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model

(X_train, y_train), (X_test, y_test) = imdb.load_data()

'''
max函数的作用是返回输入的参数中最大的参数，不论参数的类型是什么，但所有输入的参数的类型都要保持一致
'''
m = max(list(map(len, X_train)), list(map(len, X_test)))
print("最长的文本长度：", m)

'''
设定所有文本的长度为400，未满400的文本用空格填充，超过400的文本截取后400个字符，vocab_size是数据集中在字典排行最大的单词的序号+1
pad_sequences函数的作用是将列表串中的所有列表通过补零或截取变为统一大小的列表，默认是在列表前补零,截取末尾的值
max函数的作用是得到输入的list的最大值
'''
maxword = 400
X_train = sequence.pad_sequences(X_train, maxlen=maxword)
X_test = sequence.pad_sequences(X_test, maxlen=maxword)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1

'''
建立序贯模型，依次加入以下层：
嵌入层:输入词典大小，输出维度大小，输入序列长度（在需要连接扁平化和全连接层时，需要指定该选项，否则，无法计算全连接层输出的维度）
扁平层：将输入的maxword*64矩阵变为1维长度维maxword*64的向量
全连接层：2000，500，200，50，1，前四个的激活函数都是relu，最后一层使用Sigmoid，其目的是：
1，把线性函数变为非线性函数，如果只是线性函数的话，拟合结果也是线性的，效果不如加入非线性函数好
2，sigmoid函数将输入坍缩为0~1区间，我们预测的是0，1变量的概率，sigmoid函数可以计算最后输出为0或1的概率
'''
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=maxword))
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

'''
编译模型时，我们选用二维交叉熵作为损失函数，优化器选用adam，评判标准为精确度
model.summary()的作用是显示模型概况
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=100, verbose=1)
score = model.evaluate(X_test, y_test)
print(score)

model.save('SentimentModel01.hdf5')
plot_model(model, to_file='model.png', show_shapes=1)

'''
如果已训练好模型，可直接使用以下语句进行拟合数据
'''
"""
model = Sequential()
model = load_model('SentimentModel01.h5')
score = model.evaluate(X_test, y_test)
print(score)
"""

"""
下列代码主要是为了观察输入样本的特征，在实际使用中不需加入

'''
len函数的作用是，得到输入变量的长度
map函数的作用是，输入一个函数f和一个list，将该list的每一个值送入函数f中，所有得到的返回值依次构成一个新的元组
list函数的作用是将一个元组tup变为列表list
元组和列表的不同点在于：
1，元组的元素值不能修改
2，元组用小括号表示，列表用方括号表示
mean函数的作用是求列表list的平均值
'''
avg_len = list(map(len, X_train))
print(np.mean(avg_len))

'''
range函数的作用是生成整数列表，输入的参数依次为开始计数的初始值，停止技术的末尾值，计数步长，计数步长默认为1
hist函数的作用是画直方图，输入的参数依次为由要显示的变量组成的list,显示变量对应的x坐标的整数列表
'''
plt.hist(avg_len, bins=range(min(avg_len), max(avg_len) + 50, 50))
plt.show()

"""