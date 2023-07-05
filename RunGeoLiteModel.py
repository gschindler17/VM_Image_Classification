# %%
# Running all of the imported packages
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import PIL
# Notice that this import takes a while
# This is amplified if using a virtual environment
print("Beginning to import tensorflow...")
import tensorflow as tf
print("tensorflow has been imported.")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib


# %%
# Used for importing the dataset off of the web
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = "/MOUNT_HD1/gschindl/datasets/2d_shapes"

print("data_dir: {}".format(data_dir))

data_dir = pathlib.Path(data_dir).with_suffix('')
print("data_dir: {}".format(data_dir))

image_data = list(data_dir.glob('*/*.jpg'))
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Number of images found: {}".format(image_count))


# %%
# Sets parameters for the loader
batch_size = 32
img_height = 180
img_width = 180

# %%
# Beginning the splitting
# It's good practice to use a validation split when developing your model. 
# Use 80% of the images for training and 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %%
# Finding the class names from the training set
class_names = train_ds.class_names
print(class_names)



# %%
# NEW Interpret using the TensorFlow Lite Model file
TF_MODEL_FILE_PATH = '2d_lite_model.tflite' # The default path to the saved TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
seq_list = interpreter.get_signature_list()
print("seq_list: ", seq_list)


classify_lite = interpreter.get_signature_runner('serving_default')


sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('/MOUNT_HD1/gschindl/datasets/2d_shapes/Nonagon/Nonagon_a99fe1c6-2a8e-11ea-8123-8363a7ec19e6.png', origin=sunflower_url)
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

 
count = 0
table_score = [[]]
score_lite = None
for images, label in val_ds:  
  
  # print("predictions_lite:     ", predictions_lite) 
  # Specifying the correct keyword for "get_signature_runner" class
  sequential_num_kw = {seq_list['serving_default']['inputs'][0] : images}
  #print("\nsequential_num: ", sequential_num_kw)

  predictions_lite = classify_lite(**sequential_num_kw)['outputs']
  
  score_lite = tf.nn.softmax(predictions_lite)

  if count == 0:
    for i, item in enumerate(np.array(score_lite)[0]):
      table_score[0].append(item)
      # print("Special case: ", count)
  
  for r, row in enumerate(np.array(score_lite)):
    if count == 0 and (r >= (len(np.array(score_lite)) - 1)):
      # print("Should've hit break point")
      break
    table_score = np.vstack( [ table_score , row] )
    # print("Added count: ", count)

  count = count + 1


print("\ntable_score size: {} x {}".format(len(table_score), len(table_score[0])))
  
    




# %%
# NEW Confusion matrix?

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
sns.set_style('darkgrid')

classes=class_names # ordered list of class names
ytrue=[]

# TODO Should be "test_ds" instead of validation data

for images, label in val_ds:   
    for e in label:
        #print("e: ", e, " label: ", label)
        ytrue.append(classes[e]) # list of class names associated with each image file in test dataset 
ypred=[]
errors=0
count=0
preds= table_score# predict on the test data
for i, p in enumerate(preds):
    
    index=np.argmax(p) # get index of prediction with highest probability
    count +=1
    
    klass=classes[index] 
    ypred.append(klass)  
    if klass != ytrue[i]:
        errors +=1
acc= (count-errors)* 100/count
msg=f'there were {count-errors} correct predictions in {count} tests for an accuracy of {acc:6.2f} % '
print(msg) 
ypred=np.array(ypred)
ytrue=np.array(ytrue)
if len(classes)<= 30: # if more than 30 classes plot is not useful to cramed
        # create a confusion matrix 
        cm = confusion_matrix(ytrue, ypred )        
        length=len(classes)
        if length<8:
            fig_width=8
            fig_height=8
        else:
            fig_width= int(length * .5)
            fig_height= int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
        plt.xticks(np.arange(length)+.5, classes, rotation= 90)
        plt.yticks(np.arange(length)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix on Validation Data with TensorFlow Lite Model")
        plt.show()
clr = classification_report(ytrue, ypred, target_names=class_names)
print("Classification Report:\n----------------------\n", clr) 


# %%
# Visualizing the training results of the drop_model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




# %%
