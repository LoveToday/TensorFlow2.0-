import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images/255
test_images = test_images/255

ds_train_img = tf.data.Dataset.from_tensor_slices(train_images)

print(ds_train_img)

ds_train_labels = tf.data.Dataset.from_tensor_slices(train_labels)

print(ds_train_labels)

ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# 把两个数据合并
ds_train = tf.data.Dataset.zip((ds_train_img,ds_train_labels))
print(ds_train)
# 10000个数据进行乱序 repeat()无限的乱序
ds_train = ds_train.shuffle(10000).repeat().batch(64)

# 测试集
ds_test = ds_test.batch(64)

print(ds_train)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
# 因为无限循环所以 要 设置每个epochs要迭代的步数 // 取整的意思
steps_per_epochs = train_images.shape[0]//64
steps_per_epochs_test = test_images.shape[0]//64
model.fit(ds_train, epochs=5,
          steps_per_epoch=steps_per_epochs,
          validation_data=ds_test,
          validation_steps=steps_per_epochs_test)











