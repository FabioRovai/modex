# modex
[shared doc](https://docs.google.com/document/d/1lPTMbKce802x8hnFbb0dDuWAARuUbzqrVfesveHHA9s/edit )


## Step 0:
Extract details




[DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html )

[Medium](https://towardsdatascience.com/clothes-classification-with-the-deepfashion-dataset-and-fast-ai-1e174cbf0cdc)


![alt text](https://miro.medium.com/max/2000/1*jjsoLIWRNxDXOD_dui85lg.png)


[SmallDataset](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/)


![alt text](https://www.pyimagesearch.com/wp-content/uploads/2018/04/keras_multi_label_dataset.jpg)


## Step 1:
Autolabelling
![alt text](https://uploads-ssl.webflow.com/5cf12d0aeca6753441cb765c/615ae9d50080824a022e8509_SHIRT-1.png)
`tf.keras.utils.image_dataset_from_directory`
  
let's say you have 10 folders, each containing 10,000 images from a different category, and you want to train a classifier that maps an image to its category. Your training data folder would look like this:


```
  main_directory/
  ...long_sleeve/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...short_sleeve/
  ......b_image_1.jpg
  ......b_image_2.jpg
    ...V_neck/
  ......c_image_1.jpg
  ......c_image_2.jpg
  etc
  ```
  
  You may also have a validation data folder validation_data/ structured in the same way.


  
```  from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
validation_ds = image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))

model = keras.applications.Xception(weights=None, input_shape=(256, 256, 3), classes=10)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(train_ds, epochs=10, validation_data=validation_ds) 
```
  
(https://keras.io/api/preprocessing/)

## Functions:


```image_dataset_from_directory(...)```: ```Generates a tf.data.Dataset``` from image files in a directory.

```text_dataset_from_directory(...)```: ```Generates a tf.data.Dataset``` from text files in a directory.


  
  Supported image formats: jpeg, png, bmp, gif.

## Step 2:

NLP (tf-Idf and Word2Vec (?))

![alt text](https://miro.medium.com/max/1400/1*qQgnyPLDIkUmeZKN2_ZWbQ.png)


