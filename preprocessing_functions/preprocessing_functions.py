import tensorflow as tf

# MISCELLANEUS PROCESSING FUNCTIONS ()

# receives and image file and loads it as a tensor
def load_image(img_file):
  img = tf.io.read_file(img_file)
  img = tf.io.decode_jpeg(img)
  img = tf.image.resize(img, [256, 512])

  width = tf.shape(img)[1]
  width = width // 2
  #print(width)
  original_img = img[:, :width, :]
  transformed_img = img[:, width:, :]

  original_img = tf.cast(original_img, tf.float32)
  transformed_img = tf.cast(transformed_img, tf.float32)

  return original_img, transformed_img

# takes two pairs of images and resizes them
def resize(original_img, transformed_img, width, height):
  original_img = tf.image.resize(original_img, [width, height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  transformed_img = tf.image.resize(transformed_img, [width, height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return original_img, transformed_img

# takes two pair of images and normalizes them across all the color channels 
def normalize(original_img, transformed_img):
  original_img = (original_img / 127.5) - 1
  transformed_img = (transformed_img / 127.5) - 1
  return original_img, transformed_img

# receives the two pairs of images and crops them
def random_crop(original_img, transformed_img,img_width, img_height):
  stacked_img = tf.stack([original_img, transformed_img], axis = 0)
  crop_img = tf.image.random_crop(stacked_img, size = [2, img_width, img_height, 3])
  return crop_img[0], crop_img[1]


# takes a pair of images and applies reisze, cropping, and randomly flips it to increse randomness 
@tf.function()
def random_jitter(original_img, transformed_img):
  original_img, transformed_img = resize(original_img, transformed_img, 286, 286)
  original_img, transformed_img = random_crop(original_img, transformed_img)
  if tf.random.uniform(()) > 0.5:
    original_img = tf.image.flip_left_right(original_img)
    transformed_img = tf.image.flip_left_right(transformed_img)
  return original_img, transformed_img


"""
The following two functions receive an image file and loads the corresponding image pair. 
Depending on whether the image belongs to the training or testing applies a diferent kind 
of preprocessing using the functions above.
"""

def load_training_images(img_file):
  original_img,transformed_img = load_image(img_file)
  original_img,transformed_img = random_jitter(original_img,transformed_img)
  original_img,transformed_img = normalize(original_img,transformed_img)
  return original_img,transformed_img


def load_testing_images(img_file,img_width, img_height):
  original_img,transformed_img = load_image(img_file)
  original_img,transformed_img = resize(original_img,transformed_img,img_width,img_height)
  original_img,transformed_img = normalize(original_img,transformed_img)
  return original_img,transformed_img
