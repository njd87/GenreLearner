import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
#Get audio
def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

#Get label
def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2]

#Get both
def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  zero_padding = tf.zeros([1450000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=2047, frame_step=2048)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

def get_spectrogramTester(waveform):
  waveform = waveform[:160000]

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  spectrogram = tf.signal.stft(
      waveform, frame_length=1023, frame_step=500)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == genres)
  return spectrogram, label_id


def get_spectrogram_and_label_idTester(audio, label):
  spectrogram = get_spectrogramTester(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == genres)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

#Work with data
def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

def preprocess_datasetTester(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_idTester,  num_parallel_calls=AUTOTUNE)
  return output_ds

def createRanking(num, originalArrays, predictedArrays, testLabels, labels):
  for i in range(0, num):
    print("For song #{0}, predicted value is {1} with a confidence level of {2}".format(i, predictedArrays[i], originalArrays[i][predictedArrays[i]]))
    print("Original value is {0}".format(testLabels[i]))
    print("\n")



#Get directory of songs
data_dir = pathlib.Path('/Users/NJDStepinac/Desktop/SongReader2021/SongWavs1Ch')

#Get list of genres
genres = np.array(tf.io.gfile.listdir(str(data_dir)))
genres = genres[genres != '.DS_Store']

#Just in case there is any uneeded info
for genre in genres:
  if os.path.exists("/Users/NJDStepinac/Desktop/SongReader2021/SongWavs1Ch/" + genre + "/.DS_Store"):
    os.remove("/Users/NJDStepinac/Desktop/SongReader2021/SongWavs1Ch/" + genre + "/.DS_Store")
    print("Removal successful in " + genre)
  else:
    print("None in {0}".format(genre))

#Get list and randomized songs
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)

#Validation


print(len(filenames))


#Separating songs
train_files = filenames[:2748]
val_files = filenames[2748: 2748 + 393]
test_files = filenames[-393:]

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_idTester, num_parallel_calls=AUTOTUNE)

for waveform, label in spectrogram_ds.take(1):
  print(waveform.shape)

train_ds = spectrogram_ds
val_ds = preprocess_datasetTester(val_files)
test_ds = preprocess_datasetTester(test_files)

batch_size = 128
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)


#train_ds = train_ds.cache().prefetch(AUTOTUNE)
#val_ds = val_ds.cache().prefetch(AUTOTUNE)

num_labels = len(genres)


for spectrogram, labelTest in train_ds.take(1):
	input_shape = spectrogram.shape
	print(spectrogram)
	print(labelTest)
	print('Input shape:', input_shape)



norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))


for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape

print('Input shape:', input_shape)

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(200, 200), 
    norm_layer,
    layers.Conv2D(25, 5, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 2, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_labels, activation = 'softmax'),
])

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 5
history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=EPOCHS)
    #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))

#model.save('genre.model')
#model = tf.keras.models.load_model('genre.model')

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)

createRanking(20, model.predict(test_audio), y_pred, test_labels, genres)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

metrics = history.history
plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

