# modex

## Step 1:
Autolabelling
![alt text](https://uploads-ssl.webflow.com/5cf12d0aeca6753441cb765c/615ae9d50080824a022e8509_SHIRT-1.png)
`tf.keras.utils.image_dataset_from_directory`
  

Generates a `tf.data.Dataset` from image files in a directory.
  If your directory structure is:
```
  main_directory/
  ...class_a/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...class_b/
  ......b_image_1.jpg
  ......b_image_2.jpg
  ```
  
Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of images from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
  
  
  Supported image formats: jpeg, png, bmp, gif.

## Step 2:

NLP (tf-Idf and Word2Vec (?))

![alt text](https://miro.medium.com/max/1400/1*qQgnyPLDIkUmeZKN2_ZWbQ.png)


