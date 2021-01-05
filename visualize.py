import matplotlib.pyplot as plt
import tensorflow as tf

def display(display_list):
  """
  Displays the RGB image, true boundary labels, and boundaries predicted by the model

  Parameters:
  display_list - A list containing the image, label, and prediction in numpy array format
  """
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def format_output(mask, display=False):
  """
  Transform a mask (label) into numpy array format

  Parameters:
  mask - The pillow image file of the mask
  display - Adjusts pixel intensity for displaying purposes
  """
  mask = tf.reshape(mask, [-1, 128*128,2])
  if display:
    mask = tf.argmax(mask, axis=2)*255
  else:
    mask = tf.argmax(mask, axis=2)
  mask = tf.reshape(mask, [-1, 128,128,1])
  return mask