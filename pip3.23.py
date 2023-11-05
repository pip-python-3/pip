#import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Flatten,Dense

#load data
df = pd.read_csv("spam_ham_dataset.csv")

df.head()
#preprocess
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
max_len = max([len(seq) for seq in sequences])
vocab_size = len(tokenizer.word_index) + 1

X = pad_sequences(sequences, maxlen = max_len)
y = df['label_num']

#encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length = max_len))
model.add(Flatten())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 10, batch_size = 16)

_, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy', accuracy)

def predict_spam_ham(text):
  sequence = tokenizer.texts_to_sequences([text])
  sequence = pad_sequences(sequence, maxlen = max_len)
  prediction = model.predict(sequence)[0]
  if prediction >= 0.5:
    return 'spam'
  else:
    return 'ham'

texts = [
    "URGENT: You have won a cash prize of $10,000!",
    "Reminder: Your appointment is scheduled for tomorrow.",
    "Click here to claim your exclusive discount!",
    "Meeting at 2 PM in the conference room.",
]

for text in texts:
  prediction = predict_spam_ham(text)
  print("Text Sentence: ", text)
  print("Prediction: ", prediction)
  print()

"""##PRACTICAL 02 - Object detection and tracking using CNN"""

pip install ultralytics

!yolo task = detect mode = predict model = yolov8m.pt source = "/content/video.mp4"

"""##PRACTICAL 03 - Music Splitting using RNN

"""

pip install spleeter

from spleeter.separator import Separator

def seperate_audio(input_file, output_dir):
  separator = Separator('spleeter:5stems')
  separator.separate_to_file(input_file, output_dir)

input_file = '/content/song.mp3'
output_dir = '/content/Output'
seperate_audio(input_file, output_dir)

import numpy as np

#generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#add bias
X_b = np.c_[np.ones((100, 1)), X]

#set hyperparameters
learning_rate = 0.01
momentum = 0.9
n_iterations = 1000

#initialize paramaters
theta = np.random.randn(2, 1)

#Nesterov Momentum
velocity = np.zeros(theta.shape)

for iteration in range(n_iterations):
  gradients = 2 / 100 * X_b.T.dot(X_b.dot(theta) - y)
  velocity = momentum * velocity - learning_rate * gradients
  theta += momentum * velocity - learning_rate * gradients

print("Optimal Value: ", theta)

import matplotlib.pyplot as plt

#plotting og data
plt.scatter(X, y, label = 'Original Data')
plt.xlabel('X')
plt.ylabel('y')

#generating points for linear regression
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)

#Plotting linear regression line
plt.plot(X_new, y_predict, 'r-', label='Predicted Line')

plt.legend()
plt.title("Stochastic Gradient Descent with Nesterov Momentum")
plt.show()

"""##Practical 05 - NLP Application Of DL"""

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Load the IMDb dataset
vocab_size = 10000  # vocabulary size, considering the top 10,000 most common words
max_len = 200  # maximum length of reviews

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# Preprocess the data
train_data = pad_sequences(train_data, maxlen=max_len)
test_data = pad_sequences(test_data, maxlen=max_len)

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=3, batch_size=128)

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {accuracy}")

# Sample texts for prediction
sample_text = input("Enter your sentence: ")

#sample_texts = [
#    "This movie was fantastic! I loved every bit of it.",
#    "The plot was confusing and the acting was terrible.",
#    "An absolute waste of time. Would not recommend.",
#    "Great performances and a compelling story."
#]

# Preprocess the sample texts
word_index = imdb.get_word_index()
sequences = [word_index.get(word, 0) for word in sample_texts]
sequences = pad_sequences([sequences], maxlen=max_len)

# Get model predictions
predictions = model.predict(sequences)
print(f'Text: {sample_text}\nSentiment: {"Positive" if predictions > 0.5 else "Negative"} (Confidence: {predictions[0]})\n')

"""##Practical 06 - ESN"""

pip install scipy==1.7.2

pip install easyesn

from easyesn import PredictionESN
import numpy as np
import matplotlib.pyplot as plt

data_size = 1000
t = np.linspace(0, 30, data_size)
data = np.sin(t)

esn = PredictionESN(n_input = 1, n_output = 1, n_reservoir = 200)

train_length = 800
input_data = data[:train_length]
output_data = data[1:train_length+1]
esn.fit(input_data, output_data, transientTime = 100, verbose = 1)

prediction = esn.predict(input_data)

plt.figure(figsize=(12,6))
plt.plot(t[:train_length], output_data, 'r', label = "Target Signal")
plt.plot(t[:train_length], prediction, 'b--', label = "Predicted Signal")
plt.title("ESN Prediction")
plt.legend()
plt.show()

"""##Practical 07 - ICA"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

#dataset
np.random.seed(0)
n_samples = 200
time = np.linspace(0, 8, n_samples)
s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))
s3 = np.random.randn(n_samples)


S = np.c_[s1, s2, s3]

A = np.array([[1,1,1], [0.5,2,1.0], [1.5,1.0,2.0]])
X = np.dot(S, A.T)

ica = FastICA(n_components = 3)
S_ = ica.fit_transform(X)

#plot Original
plt.figure(figsize=(12,5))
plt.subplot(4, 1, 1)
plt.title("True Sources")
plt.plot(S)

#plot observed
plt.subplot(4, 1, 2)
plt.title("Observed Sources")
plt.plot(X)

#plot Seperated sources
plt.subplot(4, 1, 3)
plt.title("ICA components")
plt.plot(S_)

#plot histograms
plt.subplot(4, 1, 4)
plt.title("Histogram of ICA Components")
for i in range(3):
  plt.hist(S_[:, i], bins = 50, color = 'b', alpha = 0.5)
plt.tight_layout()
plt.show()

"""##Practical 08 - Stochastic Probabilistic Models - Cifar 10"""

pip install tensorflow

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

#load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
#normalize pixes values
train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#model compile
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

#train model
model.fit(train_images, train_labels, epochs = 10, validation_data=(test_images, test_labels))

#evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)

import numpy as np
import matplotlib.pyplot as plt

def visualize(model, images, labels, class_names):
  predictions = model.predict(images)
  num_images = images.shape[0]

  plt.figure(figsize=(15,5))
  for i in range(num_images):
    #plt.subplot(1, num_images, i+1)
    plt.imshow(images[i])
    predicted_class = np.argmax(predictions[i])
    true_class = np.argmax(labels[i])
    plt.title(f'Predicted:  {class_names[predicted_class]}\nTrue: {class_names[true_class]}')
    #plt.axis("off")
  plt.show()

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog',
               'Frog', 'Horse', 'Ship', 'Truck']

num_samples = 6
random_indices = np.random.randint(0, len(test_images), num_samples)
sample_images = test_images[random_indices]
sample_labels = test_labels[random_indices]

visualize(model, sample_images, sample_labels, class_names)
