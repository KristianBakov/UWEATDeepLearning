from keras.models import Model, load_model, model_from_json
import numpy as np
from PIL import Image
import os

# Generation resolution - Must be square 
# Training data is also scaled to this.
# Note GENERATE_RES higher than 4 will blow Google CoLab's memory.
GENERATE_RES = 2 # (1=32, 2=64, 3=96, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image 
PREVIEW_ROWS = 1
PREVIEW_COLS = 1
PREVIEW_MARGIN = 0

# Size vector to generate images from
SEED_SIZE = 250

# Configuration
DATA_PATH = './data'
OUTPUT_FOLDER = 'generated_1'

NUM_OF_IMAGES = 100

loaded_model = load_model("./saves/brick_generator.h5")
loaded_model.summary()
print("Loaded model")

def save_images(cnt,noise):
  image_array = np.full(( 
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 
      255, dtype=np.uint8)

  generated_images = loaded_model.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
    for col in range(PREVIEW_COLS):
      r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
      c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
      image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[cnt] * 255
      image_count += 1

  output_path = os.path.join(DATA_PATH, OUTPUT_FOLDER)
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  filename = os.path.join(output_path,f"image-{cnt+1}.png")
  im = Image.fromarray(image_array)
  im.save(filename)

seed = np.random.normal(0, 1, (NUM_OF_IMAGES, SEED_SIZE))
for i in range(0, NUM_OF_IMAGES):
  save_images(i,seed)

