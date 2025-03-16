import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.utils import to_categorical

(train_image, train_label), (test_image, test_label) = mnist.load_data()

print(train_image.shape, test_image.shape, train_label.shape, test_label.shape, train_label)

train_image = train_image.reshape(60000, 28 * 28)
test_image = test_image.reshape(10000, 28 * 28)

train_image = train_image.astype('float32') / 255
test_image = test_image.astype('float32') / 255

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

mlp = Sequential()
# mlp.add(Dense(units=392, input_dim=28 * 28, activation='sigmoid'))
# mlp.add(Dense(units=392, activation='sigmoid'))
mlp.add(Dense(units=512, activation='relu', input_shape=(28 * 28,)))
mlp.add(Dense(units=256, activation='relu'))
mlp.add(Dense(units=10, activation='softmax'))

mlp.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

mlp.fit(train_image, train_label, epochs=3)

loss, accuracy = mlp.evaluate(test_image, test_label, verbose=2)
print('loss:{}\naccuracy:{}'.format(loss, accuracy))
mlp.save('mnist.h5')

mlp = load_model('mnist.h5')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_labels_predict = mlp.predict(test_images.reshape(10000, 28 * 28))

img1 = test_images[100]
plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.title(test_labels_predict[100])
plt.show()
