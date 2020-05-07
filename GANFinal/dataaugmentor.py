from keras.preprocessing.image import ImageDataGenerator
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

datagen = ImageDataGenerator(
  zoom_range=[0.2,1.0],
  horizontal_flip=True,
  vertical_flip=True)

save_here = './data/augmented'
training_data_path = os.path.join('./data/brickdataset')

gen = datagen.flow_from_directory('./data/Google',target_size=(600,800),save_to_dir='./data/augmented',class_mode='binary',save_prefix='N',save_format='jpeg',batch_size=10)

for i in gen:
  idx = (gen.batch_index - 1) * gen.batch_size
  print(gen.filenames[idx : idx + gen.batch_size])