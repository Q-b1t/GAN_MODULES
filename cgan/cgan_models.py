import tensorflow as tf
from tensorflow.keras import layers

"""
These functions are oriented to building the architecture of the G.A.N
"""

# this is a single encoder block, you need as many as you need at the complete arquitecture
def u_net_encoder(filters,kernel_size,apply_batchnorm = True):
  initializer = tf.random_normal_initializer(0,0.02)
  encoder = tf.keras.Sequential()
  encoder.add(
      layers.Conv2D(
          filters,
          kernel_size,
          strides = 2,
          padding = "same",
          kernel_initializer = initializer,
          use_bias = True
      )
  )

  if apply_batchnorm:
    encoder.add(layers.BatchNormalization())
  encoder.add(layers.LeakyReLU())
  return encoder


# this is a single decoder block, you need as many as you need at the complete arquitecture
def u_net_decoder(filters,kernel_size,apply_dropout = False):
  initializer = tf.random_normal_initializer(0,0.02)
  decoder = tf.keras.Sequential()
  decoder.add(
      layers.Conv2DTranspose(
          filters,
          kernel_size,
          strides = 2,
          padding = "same",
          kernel_initializer = initializer,
          use_bias = True
      )
  )

  if apply_dropout:
    decoder.add(layers.Dropout(0.5))
  decoder.add(layers.ReLU())
  return decoder



# this is the image generator for the conditional G.A.N based on a CNN unet architecture.
def generator():
  inputs = layers.Input(
      shape = [256,256,3]
  )

  downsampling = [
      u_net_encoder(64,4,apply_batchnorm=False),
      u_net_encoder(128,4),
      u_net_encoder(256,4),
      u_net_encoder(512,4),
      u_net_encoder(512,4),
      u_net_encoder(512,4),
      u_net_encoder(512,4),
      u_net_encoder(512,4),

  ]

  upsampling = [
      u_net_decoder(512,4,apply_dropout=True),
      u_net_decoder(512,4,apply_dropout=True),
      u_net_decoder(512,4,apply_dropout=True),
      u_net_decoder(512,4),
      u_net_decoder(256,4),
      u_net_decoder(128,4),
      u_net_decoder(64,4),
  ]

  output_channels = 3
  initializer = tf.random_normal_initializer(0.,0.02)
  last = layers.Conv2DTranspose(
      output_channels,
      4,
      strides = 2,
      padding = "same",
      kernel_initializer = initializer,
      activation = "tanh"
  )

  x = inputs
  skips = list()
  for down in downsampling:
    x = down(x)
    skips.append(x)

  # reverse the skips
  skips = reversed(skips[:-1])
  """
  DA SA  -> UA  ->  UA SC
  DB SB  -> UB  ->  UB SB
  DC SC  -> UC  ->  UC SA
  """

  for up,skip in zip(upsampling,skips):
    x = up(x)
    x = layers.Concatenate()([x,skip])
  output = last(x)

  return tf.keras.Model(
      inputs = inputs,outputs = output
  )


# This is the discriminator used in the conditional G.A.N architecture
def discriminator():
  initializer = tf.random_normal_initializer(0.,0.02)

  original = layers.Input(
      shape = [256,256,3],
      name = "original_image"
  )

  transformed = layers.Input(
      shape = [256,256,3],
      name = "transformed_image"
  )

  x = layers.concatenate(
      [original,transformed]
  )

  down1 = u_net_encoder(64,4,False)(x)
  down2 = u_net_encoder(128,4)(down1)
  down3 = u_net_encoder(256,4)(down2)

  zero_pad_1 = layers.ZeroPadding2D()(down3)
  conv = layers.Conv2D(
      512,
      4,
      strides = 1,
      kernel_initializer = initializer,
      use_bias = False
  )(zero_pad_1)

  batch_norm_1 = layers.BatchNormalization()(conv)
  leaky_relu = layers.LeakyReLU()(batch_norm_1)
  zero_pad_2 = layers.ZeroPadding2D()(leaky_relu)

  last = layers.Conv2D(
      1,4,
      strides = 1,
      kernel_initializer = initializer
  )(zero_pad_2)

  return tf.keras.Model(
      inputs = [original,transformed],
      outputs = last
  )

