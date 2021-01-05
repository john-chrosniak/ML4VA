import matplotlib.pyplot as plt
import tensorflow as tf

def display(display_list):
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  print(pred_mask.shape)
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def format_output(mask, display=False):
  mask = tf.reshape(mask, [-1, 128*128,2])
  if display:
    mask = tf.argmax(mask, axis=2)*255
  else:
    mask = tf.argmax(mask, axis=2)
  mask = tf.reshape(mask, [-1, 128,128,1])
  return mask

def show_predictions(model, dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], format_output(mask[0]), format_output(create_mask(pred_mask))])
  else:
    pass