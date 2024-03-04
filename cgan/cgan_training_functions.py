import tensorflow as tf
import matplotlib.pyplot as plt 
import time
import os 
import datetime

"""
These functions are oriented to obtain loss functions and optimizers for both models
"""

# retrive the default hyperparameters in for the G.A.N according to the original publication
def get_default_gan_parameters():
  LEARNING_RATE = 2e-4
  BETA_1,BETA_2 = 0.5,999e-3
  LAMBDA = 100
  return LEARNING_RATE, BETA_1, BETA_2, LAMBDA

# the same instance of this loss functions is to be used by both the generator and the discriminator
def get_gan_loss_function():
  # the logits is because we do not pass the final logits througt an activation function in the model
  return tf.keras.losses.BinaryCrossentropy(from_logits = True) 


# loss funciton for the generator
def generator_loss(discriminator_output,generator_output,target,lambda_,loss):

  gan_loss = loss(
      tf.ones_like(discriminator_output),
      discriminator_output
  ) # step 1: compare the discriminator output with a vector of 1s

  l1_loss = tf.reduce_mean(
      tf.abs(target - generator_output)
  ) # Compute mean absolute error

  g_loss_total = gan_loss + (lambda_ * l1_loss)

  return g_loss_total,gan_loss,l1_loss

# loss funciton for the discriminator 
def discriminator_loss(d_real_output, d_generated_output,loss):
  real_loss = loss(tf.ones_like(d_real_output), d_real_output)
  generated_loss = loss(tf.zeros_like(d_generated_output), d_generated_output)
  d_total_loss = real_loss + generated_loss
  return d_total_loss


def get_model_checkpoints(gen_model,discriminator_model,generator_optimizer,discriminator_optimizer,checkpoint_dir = "./training_checkpoints"):
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                  discriminator_optimizer = discriminator_optimizer,
                                  generator = gen_model,
                                  discriminator = discriminator_model)
  return checkpoint, checkpoint_prefix

def get_file_writter(path_log = "logs/"):
  metrics = tf.summary.create_file_writer(path_log+"fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return metrics 

# get optimizers
def get_cgan_optimizers(learning_rate,beta_1,beta_2):
  generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = beta_1, beta_2 = beta_2)
  discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = beta_1, beta_2 = beta_2)
  return generator_optimizer, discriminator_optimizer

# function that uses the generator for image generation
def generate_images(model,test_input,real,step = None):
  generated_img = model(test_input,training = True)
  plt.figure(figsize=(13,7))

  img_list = [test_input[0],real[0],generated_img[0]]

  title = ["Input image","Real (Ground Truth)","Generated Image (fake)"]

  for i in range(3):
    plt.subplot(1,3,i+1)
    plt.title(title[i])
    plt.imshow(img_list[i] * 0.5 + 0.5)
    plt.axis("off")

  if step is not None:
    plt.savefig(
        f"result_pix2pix_step_{step}.png",
        bbox_inches="tight"
    )
  plt.show()



@tf.function()
def training_step(input_img, real, step,loss,gen_model,discriminator_model,generator_optimizer,discriminator_optimizer,metrics = None):
  with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
    g_output = gen_model(input_img, training = True)

    d_output_real = discriminator_model([input_img, real], training = True)
    d_output_generated = discriminator_model([input_img, g_output], training = True)

    g_total_loss, g_loss_gan, g_loss_l1 = generator_loss(d_output_generated, g_output, real,loss)
    d_loss = discriminator_loss(d_output_real, d_output_generated,loss)

  generator_gradients = g_tape.gradient(g_total_loss, gen_model.trainable_variables)
  discriminator_gradients = d_tape.gradient(d_loss, discriminator_model.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, gen_model.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_model.trainable_variables))
  if metrics is not None:
    with metrics.as_default():
      tf.summary.scalar('g_total_loss', g_total_loss, step=step//1000)
      tf.summary.scalar('g_loss_gan', g_loss_gan, step=step//1000)
      tf.summary.scalar('g_loss_l1', g_loss_l1, step=step//1000)
      tf.summary.scalar('d_loss', d_loss, step=step//1000)


def train(training_dataset, testing_dataset, steps,gen_model,discriminator_model,generator_optimizer,discriminator_optimizer,model_name='model_pix2pix.h5',checkpoint =None,checkpoint_prefix = None):
  test_input, real_input = next(iter(testing_dataset.take(1)))
  start = time.time()

  for step, (input_img, real_img) in training_dataset.repeat().take(steps).enumerate():
    if step % 1000 == 0:
      start = time.time()
      generate_images(gen_model, test_input, real_input, step)
      print(f'Step: {step//1000}K')
    training_step(input_img, real_img, step,gen_model,discriminator_model,generator_optimizer,discriminator_optimizer)
    if (step + 1) % 5000 == 0:
      if checkpoint is not None and checkpoint_prefix is not None:
        checkpoint.save(file_prefix=checkpoint_prefix)
        gen_model.save_weights(model_name)
