import tensorflow as tf
import numpy as np

# 他会把每个元素转化成tf.Tensor()数据类型
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

print(dataset)
for ele in dataset:
    print(ele)
    # 输出一个numpy数据类型
    print(ele.numpy())

# 二维列表
dataset = tf.data.Dataset.from_tensor_slices([[1,2], [3,4], [5,6]])
for ele in dataset:
    print(ele)
    print(ele.numpy())

# 使用字典来建立dataset
dataset_dic = tf.data.Dataset.from_tensor_slices({'a':[1,2,3],'b':[4,5,6],'c':[8,9,8]})

print(dataset_dic)
for ele in dataset_dic:
    print(ele)

# 此时np.array([1,2,3,4])与[1,2,3,4]效果完全相同
dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6,7,8,9]))

for ele in dataset:
    print(ele)

# 假如我们只是取出来前四个

for ele in dataset.take(2):
    print(ele)

# 如果单独使用的话
result = next(iter(dataset.take(2)))
print(result)
# tf.Tensor(1, shape=(), dtype=int64)

# 数据的乱序
dataset = dataset.shuffle(9)
# 是否重复 重复2次 并且每次都是乱序的
dataset = dataset.repeat(1)
# 添加batch 每次请求的是三个数字 与内存有关系
dataset = dataset.batch(3)
for ele in dataset:
    print(ele.numpy())


dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6,7,8,9]))
# 将每个数据都做一个平方计算

dataset = dataset.map(tf.square)

for ele in dataset:
    print(ele.numpy())











