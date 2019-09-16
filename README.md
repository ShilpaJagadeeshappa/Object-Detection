# Object-Detection
Object detection using  Machine learning(Image AI) to label ,annotate and score objects  in images and videos.We use Image AI built by Moses Olafenwa and John Olafenwa brothers, creators of TorchFusion and Authors of Introduction to Deep Computer Vision.We are extremely happy for the launch of ImageAI (v2.1.4) which is extensively used by developers/engineers/scientists/researchers across the globe.

The  field of supervised learning in AI has tremondously impacted the object tracking and detection mechanisms in many ways.We are fascinated in building out custom models and datasets of our own, improving accuracy of models, optimizing the training sessions for the operating systems. The AI concept has evolved from Alan Turing (Enigma code maker and breaker), John Mccarthy (American computer scientist and cognitive scientist) who is best known as "Father Of Artificial Intelligence" together with Marvin Minsky,Allen Newell, Herbert A.The contributions made by these great scientists has made the future path very clear. The ACM A.M. Turing Award is an annual prize given by the Association for Computing Machinery (ACM) to an individual selected for contributions "of lasting and major technical importance to the computer field". The Turing Award is generally recognized as the highest distinction in computer science and the "Nobel Prize of computing".

The Following scientists are core contributers of AI/ML-

***Scientists Contributions in AI***
- Alan J. Perlis
- Maurice Wilkes
- Richard Hamming
- Marvin Minsky
- James H. Wilkinson
- John McCarthy
- Edsger W. Dijkstra
- Charles W. Bachman
- Donald E. Knuth
- Allen Newell and Herbert A. Simon
- Michael O. Rabin
- Dana S. Scott
- John Backus
- Robert W. Floyd
- Kenneth E. Iverson
- Tony Hoare
- Edgar F. Codd
- Stephen A. Cook
- Ken Thompson and Dennis M. Ritchie
- Niklaus Wirth
- Richard M. Karp
- John Hopcroft and Robert Tarjan
- John Cocke
- Ivan Sutherland
- William Kahan
- Whitfield Diffie and Martin E. Hellman
- Tim Berners-Lee
- John L. Hennessy and David A. Patterson
- Yoshua Bengio, Geoffrey Hinton and Yann LeCun



# Introduction

ImageAI supports a list of state-of-the-art Machine Learning algorithms for image prediction, custom image prediction, object detection, video detection, video object tracking and image predictions trainings. ImageAI currently supports image prediction and training using 4 different Machine Learning algorithms trained on the ImageNet-1000 dataset. ImageAI also supports object detection, video detection and object tracking using RetinaNet, YOLOv3 and TinyYOLOv3 trained on COCO dataset. 
Eventually, ImageAI will provide support for a wider and more specialized aspects of Computer Vision including and not limited to image recognition in special environments and special fields.

ImageAI provides classes and methods for you to run image prediction your own custom objects using your own model trained with ImageAI Model Training class. You can use your custom models trained with SqueezeNet, ResNet50, InceptionV3 and DenseNet and the JSON file containing the mapping of the custom object names.

ImageAI uses the Tensorflow backbone for it's Computer Vision operations. Tensorflow supports both CPUs and GPUs ( Specifically NVIDIA GPUs. You can get one for your PC or get a PC that has one) for machine learning and artificial intelligence algorithms' implementations. 

Currently, we are using Image AI(A python library built to empower developers to build applications and systems with self-contained Deep Learning and Computer Vision capabilities using simple and few lines of code) to track objects and score onto sports platform.

In 2018, we have used the pretrained yolo v3 model to train the logos/brands and detect in sports platform or any other events.

***Dependencies***


To use ImageAI in your application developments, you must have installed the following dependencies before you install ImageAI :

1. Python 3.5.1 (and later versions) (Support for Python 2.7 coming soon
2. Tensorflow 1.4.0 (and later versions)
3. OpenCV
4. Keras 2.x
5. pip install -U tensorflow keras opencv-python

Link to download my Nike model that we have trained with 300 images of Nike logo: https://drive.google.com/file/d/1K6mscGrFylM_0kXmjsACAoVbTZn58OfB/view?usp=sharing


***Installation***

To install ImageAI, run the python installation instruction below in the command line:
pip3 install imageai --upgrade



ImageAI provides 4 different algorithms and model types to perform image prediction. To perform image prediction on any picture, take the following simple steps. The 4 algorithms provided for image prediction include SqueezeNet, ResNet, InceptionV3 and DenseNet. Each of these algorithms have individual model files which you must use depending on the choice of your algorithm. To download the model file for your choice of algorithm, click on any of the links below:

1. SqueezeNet (Size = 4.82 mb, fastest prediction time and moderate accuracy)
2. ResNet50 by Microsoft Research (Size = 98 mb, fast prediction time and high accuracy)
3. InceptionV3 by Google Brain team (Size = 91.6 mb, slow prediction time and higher accuracy)
4. DenseNet121 by Facebook AI Research (Size = 31.6 mb, slower prediction time and highest accuracy)


# Image Prediction


***FirstPrediction.py***
```

from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "1.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
    
 ```   
    




***Multiple Images Prediction***
```

from imageai.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

multiple_prediction = ImagePrediction()
multiple_prediction.setModelTypeAsResNet()
multiple_prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
multiple_prediction.loadModel()

all_images_array = []

all_files = os.listdir(execution_path)
for each_file in all_files:
    if(each_file.endswith(".jpg") or each_file.endswith(".png")):
        all_images_array.append(each_file)

results_array = multiple_prediction.predictMultipleImages(all_images_array, result_count_per_image=5)

for each_result in results_array:
    predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
    for index in range(len(predictions)):
        print(predictions[index] , " : " , percentage_probabilities[index])
    print("-----------------------")


```




***Prediction in MultiThreading***

When developing programs that run heavy task on the deafult thread like User Interfaces (UI), you should consider running your predictions in a new thread. When running image prediction using ImageAI in a new thread, you must take note the following:

You can create your prediction object, set its model type, set model path and json path outside the new thread.
The .loadModel() must be in the new thread and image prediction (predictImage()) must take place in th new thread.
Take a look of a sample code below on image prediction using multithreading:

```
from imageai.Prediction import ImagePrediction
import os
import threading

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))

picturesfolder = os.environ["USERPROFILE"] + "\\Pictures\\"
allfiles = os.listdir(picturesfolder)

class PredictionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        prediction.loadModel()
        for eachPicture in allfiles:
            if eachPicture.endswith(".png") or eachPicture.endswith(".jpg"):
                predictions, percentage_probabilities = prediction.predictImage(picturesfolder + eachPicture, result_count=1)
                for prediction, percentage_probability in zip(predictions, probabilities):
                    print(prediction , " : " , percentage_probability)

predictionThread = PredictionThread ()
predictionThread.start()

```




# ImageAI : Custom Prediction Model Training


ImageAI provides the most simple and powerful approach to training custom image prediction models using state-of-the-art SqueezeNet, ResNet50, InceptionV3 and DenseNet which you can load into the imageai.Prediction.Custom.CustomImagePrediction class. This allows you to train your own model on any set of images that corresponds to any type of objects/persons. The training process generates a JSON file that maps the objects types in your image dataset and creates lots of models. You will then pick the model with the highest accuracy and perform custom image prediction using the model and the JSON file generated.


***Custom Model Training Prediction***

1. Saving Full Custom Model (NEW)
2. Training on the IdenProf Dataset
3. Continuous Model Training (NEW)
4. Transfer Learning (Training from a pre-trained model) (NEW)


To train a custom prediction model, you need to prepare the images you want to use to train the model. You will prepare the images as follows:

1. Create a dataset folder with the name you will like your dataset to be called (e.g pets)
2. In the dataset folder, create a folder by the name train
3. In the dataset folder, create a folder by the name test
4. In the train folder, create a folder for each object you want to the model to predict and give the folder a name that corresponds to the respective object name (e.g dog, cat, squirrel, snake)
5. In the test folder, create a folder for each object you want to the model to predict and give the folder a name that corresponds to the respective object name (e.g dog, cat, squirrel, snake)
6. In each folder present in the train folder, put the images of each object in its respective folder. This images are the ones to be used to train the model To produce a model that can perform well in practical applications, I recommend you about 500 or more images per object. 1000 images per object is just great
7. In each folder present in the test folder, put about 100 to 200 images of each object in its respective folder. These images are the ones to be used to test the model as it trains
8. Once you have done this, the structure of your image dataset folder should look like below:
pets//train//dog//dog-train-images
pets//train//cat//cat-train-images
pets//train//squirrel//squirrel-train-images
pets//train//snake//snake-train-images 
pets//test//dog//dog-test-images
pets//test//cat//cat-test-images
pets//test//squirrel//squirrel-test-images
pets//test//snake//snake-test-images




#Then your training code goes as follows:
```
from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("pets")
model_trainer.trainModel(num_objects=4, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)
```
In the code above, we start the training process. The parameters stated in the function are as below:

1. num_objects : this is to state the number of object types in the image dataset
2. num_experiments : this is to state the number of times the network will train over all the training images, which is also called epochs
3. enhance_data (optional) : This is used to state if we want the network to produce modified copies of the training images for better performance.
4. batch_size : This is to state the number of images the network will process at ones. The images are processed in batches until they are exhausted per each experiment performed.
5. show_network_summary : This is to state if the network should show the structure of the training network in the console.


When you start the training, you should see something like this in the console:
```

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_2 (InputLayer)             (None, 224, 224, 3)   0
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 112, 112, 64)  9472        input_2[0][0]
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 112, 112, 64)  256         conv2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 112, 112, 64)  0           batch_normalization_1[0][0]
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 55, 55, 64)    0           activation_1[0][0]
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 55, 55, 64)    4160        max_pooling2d_1[0][0]
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 55, 55, 64)    256         conv2d_3[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 55, 55, 64)    0           batch_normalization_3[0][0]
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 55, 55, 64)    36928       activation_2[0][0]
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 55, 55, 64)    256         conv2d_4[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 55, 55, 64)    0           batch_normalization_4[0][0]
____________________________________________________________________________________________________
conv2d_5 (Conv2D)                (None, 55, 55, 256)   16640       activation_3[0][0]
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 55, 55, 256)   16640       max_pooling2d_1[0][0]
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 55, 55, 256)   1024        conv2d_5[0][0]
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 55, 55, 256)   1024        conv2d_2[0][0]
____________________________________________________________________________________________________
add_1 (Add)                      (None, 55, 55, 256)   0           batch_normalization_5[0][0]
                                                                   batch_normalization_2[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 55, 55, 256)   0           add_1[0][0]
____________________________________________________________________________________________________
conv2d_6 (Conv2D)                (None, 55, 55, 64)    16448       activation_4[0][0]
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 55, 55, 64)    256         conv2d_6[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 55, 55, 64)    0           batch_normalization_6[0][0]
____________________________________________________________________________________________________
conv2d_7 (Conv2D)                (None, 55, 55, 64)    36928       activation_5[0][0]
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 55, 55, 64)    256         conv2d_7[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 55, 55, 64)    0           batch_normalization_7[0][0]
____________________________________________________________________________________________________
conv2d_8 (Conv2D)                (None, 55, 55, 256)   16640       activation_6[0][0]
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 55, 55, 256)   1024        conv2d_8[0][0]
____________________________________________________________________________________________________
add_2 (Add)                      (None, 55, 55, 256)   0           batch_normalization_8[0][0]
                                                                   activation_4[0][0]
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 55, 55, 256)   0           add_2[0][0]
____________________________________________________________________________________________________
conv2d_9 (Conv2D)                (None, 55, 55, 64)    16448       activation_7[0][0]
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 55, 55, 64)    256         conv2d_9[0][0]
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 55, 55, 64)    0           batch_normalization_9[0][0]
____________________________________________________________________________________________________
conv2d_10 (Conv2D)               (None, 55, 55, 64)    36928       activation_8[0][0]
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 55, 55, 64)    256         conv2d_10[0][0]
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 55, 55, 64)    0           batch_normalization_10[0][0]
____________________________________________________________________________________________________
conv2d_11 (Conv2D)               (None, 55, 55, 256)   16640       activation_9[0][0]
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 55, 55, 256)   1024        conv2d_11[0][0]
____________________________________________________________________________________________________
add_3 (Add)                      (None, 55, 55, 256)   0           batch_normalization_11[0][0]
                                                                   activation_7[0][0]
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 55, 55, 256)   0           add_3[0][0]
____________________________________________________________________________________________________
conv2d_13 (Conv2D)               (None, 28, 28, 128)   32896       activation_10[0][0]
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 28, 28, 128)   512         conv2d_13[0][0]
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 28, 28, 128)   0           batch_normalization_13[0][0]
____________________________________________________________________________________________________
conv2d_14 (Conv2D)               (None, 28, 28, 128)   147584      activation_11[0][0]
____________________________________________________________________________________________________
batch_normalization_14 (BatchNor (None, 28, 28, 128)   512         conv2d_14[0][0]
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 28, 28, 128)   0           batch_normalization_14[0][0]
____________________________________________________________________________________________________
conv2d_15 (Conv2D)               (None, 28, 28, 512)   66048       activation_12[0][0]
____________________________________________________________________________________________________
conv2d_12 (Conv2D)               (None, 28, 28, 512)   131584      activation_10[0][0]
____________________________________________________________________________________________________
batch_normalization_15 (BatchNor (None, 28, 28, 512)   2048        conv2d_15[0][0]
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 28, 28, 512)   2048        conv2d_12[0][0]
____________________________________________________________________________________________________
add_4 (Add)                      (None, 28, 28, 512)   0           batch_normalization_15[0][0]
                                                                   batch_normalization_12[0][0]
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 28, 28, 512)   0           add_4[0][0]
____________________________________________________________________________________________________
conv2d_16 (Conv2D)               (None, 28, 28, 128)   65664       activation_13[0][0]
____________________________________________________________________________________________________
batch_normalization_16 (BatchNor (None, 28, 28, 128)   512         conv2d_16[0][0]
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 28, 28, 128)   0           batch_normalization_16[0][0]
____________________________________________________________________________________________________
conv2d_17 (Conv2D)               (None, 28, 28, 128)   147584      activation_14[0][0]
____________________________________________________________________________________________________
batch_normalization_17 (BatchNor (None, 28, 28, 128)   512         conv2d_17[0][0]
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 28, 28, 128)   0           batch_normalization_17[0][0]
____________________________________________________________________________________________________
conv2d_18 (Conv2D)               (None, 28, 28, 512)   66048       activation_15[0][0]
____________________________________________________________________________________________________
batch_normalization_18 (BatchNor (None, 28, 28, 512)   2048        conv2d_18[0][0]
____________________________________________________________________________________________________
add_5 (Add)                      (None, 28, 28, 512)   0           batch_normalization_18[0][0]
                                                                   activation_13[0][0]
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 28, 28, 512)   0           add_5[0][0]
____________________________________________________________________________________________________
conv2d_19 (Conv2D)               (None, 28, 28, 128)   65664       activation_16[0][0]
____________________________________________________________________________________________________
batch_normalization_19 (BatchNor (None, 28, 28, 128)   512         conv2d_19[0][0]
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 28, 28, 128)   0           batch_normalization_19[0][0]
____________________________________________________________________________________________________
conv2d_20 (Conv2D)               (None, 28, 28, 128)   147584      activation_17[0][0]
____________________________________________________________________________________________________
batch_normalization_20 (BatchNor (None, 28, 28, 128)   512         conv2d_20[0][0]
____________________________________________________________________________________________________
activation_18 (Activation)       (None, 28, 28, 128)   0           batch_normalization_20[0][0]
____________________________________________________________________________________________________
conv2d_21 (Conv2D)               (None, 28, 28, 512)   66048       activation_18[0][0]
____________________________________________________________________________________________________
batch_normalization_21 (BatchNor (None, 28, 28, 512)   2048        conv2d_21[0][0]
____________________________________________________________________________________________________
add_6 (Add)                      (None, 28, 28, 512)   0           batch_normalization_21[0][0]
                                                                   activation_16[0][0]
____________________________________________________________________________________________________
activation_19 (Activation)       (None, 28, 28, 512)   0           add_6[0][0]
____________________________________________________________________________________________________
conv2d_22 (Conv2D)               (None, 28, 28, 128)   65664       activation_19[0][0]
____________________________________________________________________________________________________
batch_normalization_22 (BatchNor (None, 28, 28, 128)   512         conv2d_22[0][0]
____________________________________________________________________________________________________
activation_20 (Activation)       (None, 28, 28, 128)   0           batch_normalization_22[0][0]
____________________________________________________________________________________________________
conv2d_23 (Conv2D)               (None, 28, 28, 128)   147584      activation_20[0][0]
____________________________________________________________________________________________________
batch_normalization_23 (BatchNor (None, 28, 28, 128)   512         conv2d_23[0][0]
____________________________________________________________________________________________________
activation_21 (Activation)       (None, 28, 28, 128)   0           batch_normalization_23[0][0]
____________________________________________________________________________________________________
conv2d_24 (Conv2D)               (None, 28, 28, 512)   66048       activation_21[0][0]
____________________________________________________________________________________________________
batch_normalization_24 (BatchNor (None, 28, 28, 512)   2048        conv2d_24[0][0]
____________________________________________________________________________________________________
add_7 (Add)                      (None, 28, 28, 512)   0           batch_normalization_24[0][0]
                                                                   activation_19[0][0]
____________________________________________________________________________________________________
activation_22 (Activation)       (None, 28, 28, 512)   0           add_7[0][0]
____________________________________________________________________________________________________
conv2d_26 (Conv2D)               (None, 14, 14, 256)   131328      activation_22[0][0]
____________________________________________________________________________________________________
batch_normalization_26 (BatchNor (None, 14, 14, 256)   1024        conv2d_26[0][0]
____________________________________________________________________________________________________
activation_23 (Activation)       (None, 14, 14, 256)   0           batch_normalization_26[0][0]
____________________________________________________________________________________________________
conv2d_27 (Conv2D)               (None, 14, 14, 256)   590080      activation_23[0][0]
____________________________________________________________________________________________________
batch_normalization_27 (BatchNor (None, 14, 14, 256)   1024        conv2d_27[0][0]
____________________________________________________________________________________________________
activation_24 (Activation)       (None, 14, 14, 256)   0           batch_normalization_27[0][0]
____________________________________________________________________________________________________
conv2d_28 (Conv2D)               (None, 14, 14, 1024)  263168      activation_24[0][0]
____________________________________________________________________________________________________
conv2d_25 (Conv2D)               (None, 14, 14, 1024)  525312      activation_22[0][0]
____________________________________________________________________________________________________
batch_normalization_28 (BatchNor (None, 14, 14, 1024)  4096        conv2d_28[0][0]
____________________________________________________________________________________________________
batch_normalization_25 (BatchNor (None, 14, 14, 1024)  4096        conv2d_25[0][0]
____________________________________________________________________________________________________
add_8 (Add)                      (None, 14, 14, 1024)  0           batch_normalization_28[0][0]
                                                                   batch_normalization_25[0][0]
____________________________________________________________________________________________________
activation_25 (Activation)       (None, 14, 14, 1024)  0           add_8[0][0]
____________________________________________________________________________________________________
conv2d_29 (Conv2D)               (None, 14, 14, 256)   262400      activation_25[0][0]
____________________________________________________________________________________________________
batch_normalization_29 (BatchNor (None, 14, 14, 256)   1024        conv2d_29[0][0]
____________________________________________________________________________________________________
activation_26 (Activation)       (None, 14, 14, 256)   0           batch_normalization_29[0][0]
____________________________________________________________________________________________________
conv2d_30 (Conv2D)               (None, 14, 14, 256)   590080      activation_26[0][0]
____________________________________________________________________________________________________
batch_normalization_30 (BatchNor (None, 14, 14, 256)   1024        conv2d_30[0][0]
____________________________________________________________________________________________________
activation_27 (Activation)       (None, 14, 14, 256)   0           batch_normalization_30[0][0]
____________________________________________________________________________________________________
conv2d_31 (Conv2D)               (None, 14, 14, 1024)  263168      activation_27[0][0]
____________________________________________________________________________________________________
batch_normalization_31 (BatchNor (None, 14, 14, 1024)  4096        conv2d_31[0][0]
____________________________________________________________________________________________________
add_9 (Add)                      (None, 14, 14, 1024)  0           batch_normalization_31[0][0]
                                                                   activation_25[0][0]
____________________________________________________________________________________________________
activation_28 (Activation)       (None, 14, 14, 1024)  0           add_9[0][0]
____________________________________________________________________________________________________
conv2d_32 (Conv2D)               (None, 14, 14, 256)   262400      activation_28[0][0]
____________________________________________________________________________________________________
batch_normalization_32 (BatchNor (None, 14, 14, 256)   1024        conv2d_32[0][0]
____________________________________________________________________________________________________
activation_29 (Activation)       (None, 14, 14, 256)   0           batch_normalization_32[0][0]
____________________________________________________________________________________________________
conv2d_33 (Conv2D)               (None, 14, 14, 256)   590080      activation_29[0][0]
____________________________________________________________________________________________________
batch_normalization_33 (BatchNor (None, 14, 14, 256)   1024        conv2d_33[0][0]
____________________________________________________________________________________________________
activation_30 (Activation)       (None, 14, 14, 256)   0           batch_normalization_33[0][0]
____________________________________________________________________________________________________
conv2d_34 (Conv2D)               (None, 14, 14, 1024)  263168      activation_30[0][0]
____________________________________________________________________________________________________
batch_normalization_34 (BatchNor (None, 14, 14, 1024)  4096        conv2d_34[0][0]
____________________________________________________________________________________________________
add_10 (Add)                     (None, 14, 14, 1024)  0           batch_normalization_34[0][0]
                                                                   activation_28[0][0]
____________________________________________________________________________________________________
activation_31 (Activation)       (None, 14, 14, 1024)  0           add_10[0][0]
____________________________________________________________________________________________________
conv2d_35 (Conv2D)               (None, 14, 14, 256)   262400      activation_31[0][0]
____________________________________________________________________________________________________
batch_normalization_35 (BatchNor (None, 14, 14, 256)   1024        conv2d_35[0][0]
____________________________________________________________________________________________________
activation_32 (Activation)       (None, 14, 14, 256)   0           batch_normalization_35[0][0]
____________________________________________________________________________________________________
conv2d_36 (Conv2D)               (None, 14, 14, 256)   590080      activation_32[0][0]
____________________________________________________________________________________________________
batch_normalization_36 (BatchNor (None, 14, 14, 256)   1024        conv2d_36[0][0]
____________________________________________________________________________________________________
activation_33 (Activation)       (None, 14, 14, 256)   0           batch_normalization_36[0][0]
____________________________________________________________________________________________________
conv2d_37 (Conv2D)               (None, 14, 14, 1024)  263168      activation_33[0][0]
____________________________________________________________________________________________________
batch_normalization_37 (BatchNor (None, 14, 14, 1024)  4096        conv2d_37[0][0]
____________________________________________________________________________________________________
add_11 (Add)                     (None, 14, 14, 1024)  0           batch_normalization_37[0][0]
                                                                   activation_31[0][0]
____________________________________________________________________________________________________
activation_34 (Activation)       (None, 14, 14, 1024)  0           add_11[0][0]
____________________________________________________________________________________________________
conv2d_38 (Conv2D)               (None, 14, 14, 256)   262400      activation_34[0][0]
____________________________________________________________________________________________________
batch_normalization_38 (BatchNor (None, 14, 14, 256)   1024        conv2d_38[0][0]
____________________________________________________________________________________________________
activation_35 (Activation)       (None, 14, 14, 256)   0           batch_normalization_38[0][0]
____________________________________________________________________________________________________
conv2d_39 (Conv2D)               (None, 14, 14, 256)   590080      activation_35[0][0]
____________________________________________________________________________________________________
batch_normalization_39 (BatchNor (None, 14, 14, 256)   1024        conv2d_39[0][0]
____________________________________________________________________________________________________
activation_36 (Activation)       (None, 14, 14, 256)   0           batch_normalization_39[0][0]
____________________________________________________________________________________________________
conv2d_40 (Conv2D)               (None, 14, 14, 1024)  263168      activation_36[0][0]
____________________________________________________________________________________________________
batch_normalization_40 (BatchNor (None, 14, 14, 1024)  4096        conv2d_40[0][0]
____________________________________________________________________________________________________
add_12 (Add)                     (None, 14, 14, 1024)  0           batch_normalization_40[0][0]
                                                                   activation_34[0][0]
____________________________________________________________________________________________________
activation_37 (Activation)       (None, 14, 14, 1024)  0           add_12[0][0]
____________________________________________________________________________________________________
conv2d_41 (Conv2D)               (None, 14, 14, 256)   262400      activation_37[0][0]
____________________________________________________________________________________________________
batch_normalization_41 (BatchNor (None, 14, 14, 256)   1024        conv2d_41[0][0]
____________________________________________________________________________________________________
activation_38 (Activation)       (None, 14, 14, 256)   0           batch_normalization_41[0][0]
____________________________________________________________________________________________________
conv2d_42 (Conv2D)               (None, 14, 14, 256)   590080      activation_38[0][0]
____________________________________________________________________________________________________
batch_normalization_42 (BatchNor (None, 14, 14, 256)   1024        conv2d_42[0][0]
____________________________________________________________________________________________________
activation_39 (Activation)       (None, 14, 14, 256)   0           batch_normalization_42[0][0]
____________________________________________________________________________________________________
conv2d_43 (Conv2D)               (None, 14, 14, 1024)  263168      activation_39[0][0]
____________________________________________________________________________________________________
batch_normalization_43 (BatchNor (None, 14, 14, 1024)  4096        conv2d_43[0][0]
____________________________________________________________________________________________________
add_13 (Add)                     (None, 14, 14, 1024)  0           batch_normalization_43[0][0]
                                                                   activation_37[0][0]
____________________________________________________________________________________________________
activation_40 (Activation)       (None, 14, 14, 1024)  0           add_13[0][0]
____________________________________________________________________________________________________
conv2d_45 (Conv2D)               (None, 7, 7, 512)     524800      activation_40[0][0]
____________________________________________________________________________________________________
batch_normalization_45 (BatchNor (None, 7, 7, 512)     2048        conv2d_45[0][0]
____________________________________________________________________________________________________
activation_41 (Activation)       (None, 7, 7, 512)     0           batch_normalization_45[0][0]
____________________________________________________________________________________________________
conv2d_46 (Conv2D)               (None, 7, 7, 512)     2359808     activation_41[0][0]
____________________________________________________________________________________________________
batch_normalization_46 (BatchNor (None, 7, 7, 512)     2048        conv2d_46[0][0]
____________________________________________________________________________________________________
activation_42 (Activation)       (None, 7, 7, 512)     0           batch_normalization_46[0][0]
____________________________________________________________________________________________________
conv2d_47 (Conv2D)               (None, 7, 7, 2048)    1050624     activation_42[0][0]
____________________________________________________________________________________________________
conv2d_44 (Conv2D)               (None, 7, 7, 2048)    2099200     activation_40[0][0]
____________________________________________________________________________________________________
batch_normalization_47 (BatchNor (None, 7, 7, 2048)    8192        conv2d_47[0][0]
____________________________________________________________________________________________________
batch_normalization_44 (BatchNor (None, 7, 7, 2048)    8192        conv2d_44[0][0]
____________________________________________________________________________________________________
add_14 (Add)                     (None, 7, 7, 2048)    0           batch_normalization_47[0][0]
                                                                   batch_normalization_44[0][0]
____________________________________________________________________________________________________
activation_43 (Activation)       (None, 7, 7, 2048)    0           add_14[0][0]
____________________________________________________________________________________________________
conv2d_48 (Conv2D)               (None, 7, 7, 512)     1049088     activation_43[0][0]
____________________________________________________________________________________________________
batch_normalization_48 (BatchNor (None, 7, 7, 512)     2048        conv2d_48[0][0]
____________________________________________________________________________________________________
activation_44 (Activation)       (None, 7, 7, 512)     0           batch_normalization_48[0][0]
____________________________________________________________________________________________________
conv2d_49 (Conv2D)               (None, 7, 7, 512)     2359808     activation_44[0][0]
____________________________________________________________________________________________________
batch_normalization_49 (BatchNor (None, 7, 7, 512)     2048        conv2d_49[0][0]
____________________________________________________________________________________________________
activation_45 (Activation)       (None, 7, 7, 512)     0           batch_normalization_49[0][0]
____________________________________________________________________________________________________
conv2d_50 (Conv2D)               (None, 7, 7, 2048)    1050624     activation_45[0][0]
____________________________________________________________________________________________________
batch_normalization_50 (BatchNor (None, 7, 7, 2048)    8192        conv2d_50[0][0]
____________________________________________________________________________________________________
add_15 (Add)                     (None, 7, 7, 2048)    0           batch_normalization_50[0][0]
                                                                   activation_43[0][0]
____________________________________________________________________________________________________
activation_46 (Activation)       (None, 7, 7, 2048)    0           add_15[0][0]
____________________________________________________________________________________________________
conv2d_51 (Conv2D)               (None, 7, 7, 512)     1049088     activation_46[0][0]
____________________________________________________________________________________________________
batch_normalization_51 (BatchNor (None, 7, 7, 512)     2048        conv2d_51[0][0]
____________________________________________________________________________________________________
activation_47 (Activation)       (None, 7, 7, 512)     0           batch_normalization_51[0][0]
____________________________________________________________________________________________________
conv2d_52 (Conv2D)               (None, 7, 7, 512)     2359808     activation_47[0][0]
____________________________________________________________________________________________________
batch_normalization_52 (BatchNor (None, 7, 7, 512)     2048        conv2d_52[0][0]
____________________________________________________________________________________________________
activation_48 (Activation)       (None, 7, 7, 512)     0           batch_normalization_52[0][0]
____________________________________________________________________________________________________
conv2d_53 (Conv2D)               (None, 7, 7, 2048)    1050624     activation_48[0][0]
____________________________________________________________________________________________________
batch_normalization_53 (BatchNor (None, 7, 7, 2048)    8192        conv2d_53[0][0]
____________________________________________________________________________________________________
add_16 (Add)                     (None, 7, 7, 2048)    0           batch_normalization_53[0][0]
                                                                   activation_46[0][0]
____________________________________________________________________________________________________
activation_49 (Activation)       (None, 7, 7, 2048)    0           add_16[0][0]
____________________________________________________________________________________________________
global_avg_pooling (GlobalAverag (None, 2048)          0           activation_49[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 10)            20490       global_avg_pooling[0][0]
____________________________________________________________________________________________________
activation_50 (Activation)       (None, 10)            0           dense_1[0][0]
====================================================================================================
Total params: 23,608,202
Trainable params: 23,555,082
Non-trainable params: 53,120
____________________________________________________________________________________________________
Using Enhanced Data Generation
Found 4000 images belonging to 4 classes.
Found 800 images belonging to 4 classes.
JSON Mapping for the model classes saved to  C:\Users\User\PycharmProjects\ImageAITest\pets\json\model_class.json
Number of experiments (Epochs) :  100
When the training progress progresses, you will see results as follows in the console:

Epoch 1/100
 1/25 [>.............................] - ETA: 52s - loss: 2.3026 - acc: 0.2500
 2/25 [=>............................] - ETA: 41s - loss: 2.3027 - acc: 0.1250
 3/25 [==>...........................] - ETA: 37s - loss: 2.2961 - acc: 0.1667
 4/25 [===>..........................] - ETA: 36s - loss: 2.2980 - acc: 0.1250
 5/25 [=====>........................] - ETA: 33s - loss: 2.3178 - acc: 0.1000
 6/25 [======>.......................] - ETA: 31s - loss: 2.3214 - acc: 0.0833
 7/25 [=======>......................] - ETA: 30s - loss: 2.3202 - acc: 0.0714
 8/25 [========>.....................] - ETA: 29s - loss: 2.3207 - acc: 0.0625
 9/25 [=========>....................] - ETA: 27s - loss: 2.3191 - acc: 0.0556
10/25 [===========>..................] - ETA: 25s - loss: 2.3167 - acc: 0.0750
11/25 [============>.................] - ETA: 23s - loss: 2.3162 - acc: 0.0682
12/25 [=============>................] - ETA: 21s - loss: 2.3143 - acc: 0.0833
13/25 [==============>...............] - ETA: 20s - loss: 2.3135 - acc: 0.0769
14/25 [===============>..............] - ETA: 18s - loss: 2.3132 - acc: 0.0714
15/25 [=================>............] - ETA: 16s - loss: 2.3128 - acc: 0.0667
16/25 [==================>...........] - ETA: 15s - loss: 2.3121 - acc: 0.0781
17/25 [===================>..........] - ETA: 13s - loss: 2.3116 - acc: 0.0735
18/25 [====================>.........] - ETA: 12s - loss: 2.3114 - acc: 0.0694
19/25 [=====================>........] - ETA: 10s - loss: 2.3112 - acc: 0.0658
20/25 [=======================>......] - ETA: 8s - loss: 2.3109 - acc: 0.0625
21/25 [========================>.....] - ETA: 7s - loss: 2.3107 - acc: 0.0595
22/25 [=========================>....] - ETA: 5s - loss: 2.3104 - acc: 0.0568
23/25 [==========================>...] - ETA: 3s - loss: 2.3101 - acc: 0.0543
24/25 [===========================>..] - ETA: 1s - loss: 2.3097 - acc: 0.0625Epoch 00000: saving model to C:\Users\Moses\Documents\Moses\W7\AI\Custom Datasets\IDENPROF\idenprof-small-test\idenprof\models\model_ex-000_acc-0.100000.h5

25/25 [==============================] - 51s - loss: 2.3095 - acc: 0.0600 - val_loss: 2.3026 - val_acc: 0.1000


```


***Continuous Model Training***


ImageAI now allows you to continue training your custom model on your previously saved model. This is useful in cases of incomplete training due compute time limits/large size of dataset or should you intend to further train your model.Kindly note that continuous training is for using a previously saved model to train on the same dataset the model was trained on. All you need to do is specify the continue_from_model parameter to the path of the previously saved model in your trainModel() function. See an example code below.

```
from imageai.Prediction.Custom import ModelTraining
import os

trainer = ModelTraining()
trainer.setModelTypeAsDenseNet()
trainer.setDataDirectory("idenprof")
trainer.trainModel(num_objects=10, num_experiments=50, enhance_data=True, batch_size=8, show_network_summary=True, continue_from_model="idenprof_densenet-0.763500.h5")
```





***Transfer Learning (Training from a pre-trained model)***
```
from imageai.Prediction.Custom import ModelTraining
import os

trainer = ModelTraining()
trainer.setModelTypeAsResNet()
trainer.setDataDirectory("idenprof")
trainer.trainModel(num_objects=10, num_experiments=50, enhance_data=True, batch_size=32, show_network_summary=True,transfer_from_model="resnet50_weights_tf_dim_ordering_tf_kernels.h5", initial_num_objects=1000)

```



***Prediction Speed***
ImageAI now provides prediction speeds for all image prediction tasks. The prediction speeds allow you to reduce the time of prediction at a rate between 20% - 60%, and yet having just slight changes but accurate prediction results. The available prediction speeds are "normal"(default), "fast", "faster" and "fastest". All you need to do is to state the speed mode you desire when loading the model as seen below.


***prediction.loadModel(prediction_speed="fast")***

To observe the differences in the prediction speeds, look below for each speed applied to multiple prediction with time taken to predict and predictions given. The results below are obtained from predictions performed on a Windows 8 laptop with Intel Celeron N2820 CPU, with processor speed of 2.13GHz

***Prediction Speed = "normal" , Prediction Time = 5.9 seconds***
```
convertible : 52.459555864334106
sports_car : 37.61284649372101
pickup : 3.1751200556755066
car_wheel : 1.817505806684494
minivan : 1.7487050965428352
-----------------------
toilet_tissue : 13.99008333683014
jeep : 6.842949986457825
car_wheel : 6.71963095664978
seat_belt : 6.704962253570557
minivan : 5.861184373497963
-----------------------
bustard : 52.03368067741394
vulture : 20.936034619808197
crane : 10.620515048503876
kite : 10.20539253950119
white_stork : 1.6472270712256432


```

***Prediction Speed = "fast" , Prediction Time = 3.4 seconds***
```
sports_car : 55.5136501789093
pickup : 19.860029220581055
convertible : 17.88402795791626
tow_truck : 2.357563190162182
car_wheel : 1.8646160140633583
-----------------------
drum : 12.241223454475403
toilet_tissue : 10.96322312951088
car_wheel : 10.776633024215698
dial_telephone : 9.840480983257294
toilet_seat : 8.989936858415604
-----------------------
vulture : 52.81011462211609
bustard : 45.628002285957336
kite : 0.8065823465585709
goose : 0.3629807382822037
crane : 0.21266008261591196
-----------------------


```

***Prediction Speed = "faster" , Prediction Time = 2.7 seconds***

```

sports_car : 79.90474104881287
tow_truck : 9.751049429178238
convertible : 7.056044787168503
racer : 1.8735893070697784
car_wheel : 0.7379394955933094
-----------------------
oil_filter : 73.52778315544128
jeep : 11.926891654729843
reflex_camera : 7.9965077340602875
Polaroid_camera : 0.9798810817301273
barbell : 0.8661789819598198
-----------------------
vulture : 93.00530552864075
bustard : 6.636220961809158
kite : 0.15161558985710144
bald_eagle : 0.10513027664273977
crane : 0.05982434959150851
----------------------

```

### ImageAI : Video Object Detection, Tracking and Analysis
```
-First Video Object Detection
-Custom Video Object Detection (Object Tracking)
-Camera / Live Stream Video Detection
-Video Analysis
-Detection Speed
- Hiding/Showing Object Name and Probability
-Frame Detection Intervals
-Video Detection Timeout (NEW)
```

ImageAI provides convenient, flexible and powerful methods to perform object detection on videos. The video object detection class provided only supports RetinaNet, YOLOv3 and TinyYOLOv3. This version of ImageAI provides commercial grade video objects detection features, which include but not limited to device/IP camera inputs, per frame, per second, per minute and entire video analysis for storing in databases and/or real-time visualizations and for future insights.
```
-RetinaNet (Size = 145 mb, high performance and accuracy, with longer detection time)
-YOLOv3 (Size = 237 mb, moderate performance and accuracy, with a moderate detection time)
-TinyYOLOv3 (Size = 34 mb, optimized for speed and moderate performance, with fast detection time)
```

Because video object detection is a compute intensive tasks, we advise you perform this experiment using a computer with a NVIDIA GPU and the GPU version of Tensorflow installed. Performing Video Object Detection CPU will be slower than using an NVIDIA GPU powered computer. You can use Google Colab for this experiment as it has an NVIDIA K80 GPU available for free.

Once you download the object detection model file, you should copy the model file to the your project folder where your .py files will be. Then create a python file and give it a name; an example is FirstVideoObjectDetection.py. Then write the code below into the python file

***FirstVideoObjectDetection.py***

```
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_detected")
                                , frames_per_second=20, log_progress=True)
print(video_path)
```

Interestingly, ImageAI allow you to perform detection for one or more of the items above. That means you can customize the type of object(s) you want to be detected in the video. Let's take a look at the code below:

```
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True)

video_path = detector.detectCustomObjectsFromVideo(
                custom_objects=custom_objects,
                input_file_path=os.path.join(execution_path, "traffic.mp4"),
                output_file_path=os.path.join(execution_path, "traffic_custom_detected"),
                frames_per_second=20, log_progress=True)
print(video_path)
```

***Camera / Live Stream Video Detection***

ImageAI now allows live-video detection with support for camera inputs. Using OpenCV's VideoCapture() function, you can load live-video streams from a device camera, cameras connected by cable or IP cameras, and parse it into ImageAI's detectObjectsFromVideo() and detectCustomObjectsFromVideo() functions. All features that are supported for detecting objects in a video file is also available for detecting objects in a camera's live-video feed. Find below an example of detecting live-video feed from the device camera.

```
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()


camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


video_path = detector.detectObjectsFromVideo(
                camera_input=camera,
                output_file_path=os.path.join(execution_path, "camera_detected_video"),
                frames_per_second=20, log_progress=True, minimum_percentage_probability=40)
                
```


###Video Analysis


ImageAI now provide commercial-grade video analysis in the Video Object Detection class, for both video file inputs and camera inputs. This feature allows developers to obtain deep insights into any video processed with ImageAI. This insights can be visualized in real-time, stored in a NoSQL database for future review or analysis.

For video analysis, the detectObjectsFromVideo() and detectCustomObjectsFromVideo() now allows you to state your own defined functions which will be executed for every frame, seconds and/or minute of the video detected as well as a state a function that will be executed at the end of a video detection. Once this functions are stated, they will receive raw but comprehensive analytical data on the index of the frame/second/minute, objects detected (name, percentage_probability and box_points), number of instances of each unique object detected and average number of occurrence of each unique object detected over a second/minute and entire video.

To obtain the video analysis, all you need to do is specify a function, state the corresponding parameters it will be receiving and parse the function name into the per_frame_function, per_second_function, per_minute_function and video_complete_function parameters in the detection function. Find below examples of video analysis functions.


```

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
video_detector.loadModel()

video_detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "traffic.mp4"),
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    frames_per_second=10,
    per_second_function=forSeconds,
    per_frame_function=forFrame,
    per_minute_function=forMinute,
    minimum_percentage_probability=30
)

```

When the detection starts on a video feed, be it from a video file or camera input, the result will have the format as below:

Results for the Frame function

```
FOR FRAME : 1
 
Output for each object : [{'box_points': (362, 295, 443, 355), 'name': 'boat', 'percentage_probability': 26.666194200515747}, {'box_points': (319, 245, 386, 296), 'name': 'boat', 'percentage_probability': 30.052968859672546}, {'box_points': (219, 308, 341, 358), 'name': 'boat', 'percentage_probability': 47.46982455253601}, {'box_points': (589, 198, 621, 241), 'name': 'bus', 'percentage_probability': 24.62330162525177}, {'box_points': (519, 181, 583, 263), 'name': 'bus', 'percentage_probability': 27.446213364601135}, {'box_points': (493, 197, 561, 272), 'name': 'bus', 'percentage_probability': 59.81815457344055}, {'box_points': (432, 187, 491, 240), 'name': 'bus', 'percentage_probability': 64.42965269088745}, {'box_points': (157, 225, 220, 255), 'name': 'car', 'percentage_probability': 21.150341629981995}, {'box_points': (324, 249, 377, 293), 'name': 'car', 'percentage_probability': 24.089913070201874}, {'box_points': (152, 275, 260, 327), 'name': 'car', 'percentage_probability': 30.341443419456482}, {'box_points': (433, 198, 485, 244), 'name': 'car', 'percentage_probability': 37.205660343170166}, {'box_points': (184, 226, 233, 260), 'name': 'car', 'percentage_probability': 38.52525353431702}, {'box_points': (3, 296, 134, 359), 'name': 'car', 'percentage_probability': 47.80363142490387}, {'box_points': (357, 302, 439, 359), 'name': 'car', 'percentage_probability': 47.94844686985016}, {'box_points': (481, 266, 546, 314), 'name': 'car', 'percentage_probability': 65.8585786819458}, {'box_points': (597, 269, 624, 318), 'name': 'person', 'percentage_probability': 27.125394344329834}]
 
Output count for unique objects : {'bus': 4, 'boat': 3, 'person': 1, 'car': 8}

------------END OF A FRAME --------------
```


For any function you parse into the per_frame_function, the function will be executed after every single video frame is processed and he following will be parsed into it:

-Frame Index: This is the position number of the frame inside the video (e.g 1 for first frame and 20 for twentieth frame).
-Output Array: This is an array of dictionaries. Each dictionary corresponds to each detected object in the image and it contains the "name", "percentage_probabaility" and "box_points"(x1,y1,x2,y2) values of the object.
-Output Count: This is a dictionary that has the name of each unique object detected as its keys and the number of instances of the objects detected as the values.
Results for the Second function

```
FOR SECOND : 1
 
 Array for the outputs of each frame [[{'box_points': (362, 295, 443, 355), 'name': 'boat', 'percentage_probability': 26.666194200515747}, {'box_points': (319, 245, 386, 296), 'name': 'boat', 'percentage_probability': 30.052968859672546}, {'box_points': (219, 308, 341, 358), 'name': 'boat', 'percentage_probability': 47.46982455253601}, {'box_points': (589, 198, 621, 241), 'name': 'bus', 'percentage_probability': 24.62330162525177}, {'box_points': (519, 181, 583, 263), 'name': 'bus', 'percentage_probability': 27.446213364601135}, {'box_points': (493, 197, 561, 272), 'name': 'bus', 'percentage_probability': 59.81815457344055}, {'box_points': (432, 187, 491, 240), 'name': 'bus', 'percentage_probability': 64.42965269088745}, {'box_points': (157, 225, 220, 255), 'name': 'car', 'percentage_probability': 21.150341629981995}, {'box_points': (324, 249, 377, 293), 'name': 'car', 'percentage_probability': 24.089913070201874}, {'box_points': (152, 275, 260, 327), 'name': 'car', 'percentage_probability': 30.341443419456482}, {'box_points': (433, 198, 485, 244), 'name': 'car', 'percentage_probability': 37.205660343170166}, {'box_points': (184, 226, 233, 260), 'name': 'car', 'percentage_probability': 38.52525353431702}, {'box_points': (3, 296, 134, 359), 'name': 'car', 'percentage_probability': 47.80363142490387}, {'box_points': (357, 302, 439, 359), 'name': 'car', 'percentage_probability': 47.94844686985016}, {'box_points': (481, 266, 546, 314), 'name': 'car', 'percentage_probability': 65.8585786819458}, {'box_points': (597, 269, 624, 318), 'name': 'person', 'percentage_probability': 27.125394344329834}],
 [{'box_points': (316, 240, 384, 302), 'name': 'boat', 'percentage_probability': 29.594269394874573}, {'box_points': (361, 295, 441, 354), 'name': 'boat', 'percentage_probability': 36.11513376235962}, {'box_points': (216, 305, 340, 357), 'name': 'boat', 'percentage_probability': 44.89373862743378}, {'box_points': (432, 198, 488, 244), 'name': 'truck', 'percentage_probability': 22.914741933345795}, {'box_points': (589, 199, 623, 240), 'name': 'bus', 'percentage_probability': 20.545457303524017}, {'box_points': (519, 182, 583, 263), 'name': 'bus', 'percentage_probability': 24.467085301876068}, {'box_points': (492, 197, 563, 271), 'name': 'bus', 'percentage_probability': 61.112016439437866}, {'box_points': (433, 188, 490, 241), 'name': 'bus', 'percentage_probability': 65.08989334106445}, {'box_points': (352, 303, 442, 357), 'name': 'car', 'percentage_probability': 20.025095343589783}, {'box_points': (136, 172, 188, 195), 'name': 'car', 'percentage_probability': 21.571354568004608}, {'box_points': (152, 276, 261, 326), 'name': 'car', 'percentage_probability': 33.07966589927673}, {'box_points': (181, 225, 230, 256), 'name': 'car', 'percentage_probability': 35.111838579177856}, {'box_points': (432, 198, 488, 244), 'name': 'car', 'percentage_probability': 36.25282347202301}, {'box_points': (3, 292, 130, 360), 'name': 'car', 'percentage_probability': 67.55480170249939}, {'box_points': (479, 265, 546, 314), 'name': 'car', 'percentage_probability': 71.47912979125977}, {'box_points': (597, 269, 625, 318), 'name': 'person', 'percentage_probability': 25.903674960136414}],................, 
[{'box_points': (133, 250, 187, 278), 'name': 'umbrella', 'percentage_probability': 21.518094837665558}, {'box_points': (154, 233, 218, 259), 'name': 'umbrella', 'percentage_probability': 23.687003552913666}, {'box_points': (348, 311, 425, 360), 'name': 'boat', 'percentage_probability': 21.015766263008118}, {'box_points': (11, 164, 137, 225), 'name': 'bus', 'percentage_probability': 32.20453858375549}, {'box_points': (424, 187, 485, 243), 'name': 'bus', 'percentage_probability': 38.043853640556335}, {'box_points': (496, 186, 570, 264), 'name': 'bus', 'percentage_probability': 63.83994221687317}, {'box_points': (588, 197, 622, 240), 'name': 'car', 'percentage_probability': 23.51653128862381}, {'box_points': (58, 268, 111, 303), 'name': 'car', 'percentage_probability': 24.538707733154297}, {'box_points': (2, 246, 72, 301), 'name': 'car', 'percentage_probability': 28.433072566986084}, {'box_points': (472, 273, 539, 323), 'name': 'car', 'percentage_probability': 87.17672824859619}, {'box_points': (597, 270, 626, 317), 'name': 'person', 'percentage_probability': 27.459821105003357}]
 ]
 
Array for output count for unique objects in each frame : [{'bus': 4, 'boat': 3, 'person': 1, 'car': 8},
 {'truck': 1, 'bus': 4, 'boat': 3, 'person': 1, 'car': 7},
 {'bus': 5, 'boat': 2, 'person': 1, 'car': 5},
 {'bus': 5, 'boat': 1, 'person': 1, 'car': 9},
 {'truck': 1, 'bus': 2, 'car': 6, 'person': 1},
 {'truck': 2, 'bus': 4, 'boat': 2, 'person': 1, 'car': 7},
 {'truck': 1, 'bus': 3, 'car': 7, 'person': 1, 'umbrella': 1},
 {'bus': 4, 'car': 7, 'person': 1, 'umbrella': 2},
 {'bus': 3, 'car': 6, 'boat': 1, 'person': 1, 'umbrella': 3},
 {'bus': 3, 'car': 4, 'boat': 1, 'person': 1, 'umbrella': 2}]
 
Output average count for unique objects in the last second: {'truck': 0.5, 'bus': 3.7, 'umbrella': 0.8, 'boat': 1.3, 'person': 1.0, 'car': 6.6}

------------END OF A SECOND --------------
```

In the above result, the video was processed and saved in 10 frames per second (FPS). For any function you parse into the per_second_function, the function will be executed after every single second of the video that is processed and he following will be parsed into it:
```
-Second Index: This is the position number of the second inside the video (e.g 1 for first second and 20 for twentieth second).
-Output Array: This is an array of arrays, with each contained array and its position (array index + 1) corresponding to the equivalent frame in the last second of the video (In the above example, their are 10 arrays which corresponds to the 10 frames contained in one second). Each contained array contains dictionaries. Each dictionary corresponds to each detected object in the image and it contains the "name", "percentage_probabaility" and "box_points"(x1,y1,x2,y2) values of the object.
-Count arrays: This is an array of dictionaries. Each dictionary and its position (array index + 1) corresponds to the equivalent frame in the last second of he video. Each dictionary has the name of each unique object detected as its keys and the number of instances of the objects detected as the values.
-Average Output Count: This is a dictionary that has the name of each unique object detected in the last second as its keys and the average number of instances of the objects detected across the number of frames as the values.
```
Results for the Minute function The above set of 4 parameters that are returned for every second of the video processed is the same parameters to that will be returned for every minute of the video processed. The difference is that the index returned corresponds to the minute index, the output_arrays is an array that contains the number of FPS * 60 number of arrays (in the code example above, 10 frames per second(fps) * 60 seconds = 600 frames = 600 arrays), and the count_arrays is an array that contains the number of FPS * 60 number of dictionaries (in the code example above, 10 frames per second(fps) * 60 seconds = 600 frames = 600 dictionaries) and the average_output_count is a dictionary that covers all the objects detected in all the frames contained in the last minute.


Results for the Video Complete Function ImageAI allows you to obtain complete analysis of the entire video processed. All you need is to define a function like the forSecond or forMinute function and set the video_complete_function parameter into your .detectObjectsFromVideo() or .detectCustomObjectsFromVideo() function. The same values for the per_second-function and per_minute_function will be returned. The difference is that no index will be returned and the other 3 values will be returned, and the 3 values will cover all frames in the video. Below is a sample function:

```
def forFull(output_arrays, count_arrays, average_output_count):
    #Perform action on the 3 parameters returned into the function

video_detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "traffic.mp4"),
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    frames_per_second=10,
    video_complete_function=forFull,
    minimum_percentage_probability=30
)
```

FINAL NOTE ON VIDEO ANALYSIS : ImageAI allows you to obtain the detected video frame as a Numpy array at each frame, second and minute function. All you need to do is specify one more parameter in your function and set return_detected_frame=True in your detectObjectsFromVideo() or detectCustomObjectsFrom() function. Once this is set, the extra parameter you sepecified in your function will be the Numpy array of the detected frame. See a sample below:

```
def forFrame(frame_number, output_array, output_count, detected_frame):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
	print("Returned Objects is : ", type(detected_frame))
    print("------------END OF A FRAME --------------")

video_detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "traffic.mp4"),
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    frames_per_second=10,
    per_frame_function=forFrame,
    minimum_percentage_probability=30,
    return_detected_frame=True
)
```
***Video Detection Speed***

ImageAI now provides detection speeds for all video object detection tasks. The detection speeds allow you to reduce the time of detection at a rate between 20% - 80%, and yet having just slight changes but accurate detection results. Coupled with lowering the minimum_percentage_probability parameter, detections can closely match the normal speed and yet reduce detection time drastically. The available detection speeds are "normal"(default), "fast", "faster" , "fastest" and "flash". All you need to do is to state the speed mode you desire when loading the model as seen below.

detector.loadModel(detection_speed="fast")

***Frame Detection Intervals***

The above video objects detection task are optimized for frame-real-time object detections that ensures that objects in every frame of the video is detected. ImageAI provides you the option to adjust the video frame detections which can speed up your video detection process. When calling the .detectObjectsFromVideo() or .detectCustomObjectsFromVideo(), you can specify at which frame interval detections should be made. By setting the frame_detection_interval parameter to be equal to 5 or 20, that means the object detections in the video will be updated after 5 frames or 20 frames. If your output video frames_per_second is set to 20, that means the object detections in the video will be updated once in every quarter of a second or every second. This is useful in case scenarious where the available compute is less powerful and speeds of moving objects are low. This ensures you can have objects detected as second-real-time , half-a-second-real-time or whichever way suits your needs. We conducted video object detection on the same input video we have been using all this while by applying a frame_detection_interval value equal to 5. The results below are obtained from detections performed on a NVIDIA K80 GPU. See the results and link to download the videos below:

```
Link1: https://drive.google.com/file/d/10m6kXlXWGOGc-IPw6TsKxBi-SXXOH9xK/view
Link2: https://drive.google.com/open?id=17934YONVSXvd4uuJE0KwenEFks7fFYe4
Link3: https://drive.google.com/open?id=1cs_06CuhXDvZp3fHJWFpam-31eclOhc-
```

***Video Detection Timeout***

ImageAI now allows you to set a timeout in seconds for detection of objects in videos or camera live feed. To set a timeout for your video detection code, all you need to do is specify the detection_timeout parameter in the detectObjectsFromVideo() function to the number of desired seconds. In the example code below, we set detection_timeout to 120 seconds (2 minutes).


```
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                             output_file_path=os.path.join(execution_path, "camera_detected_video"),
                                             frames_per_second=20,
                                             log_progress=True,
                                             minimum_percentage_probability=40,
                                             detection_timeout=120)
                                             
 ```

# ImageAI : Custom Detection Model Training

ImageAI provides the most simple and powerful approach to training custom object detection models using the YOLOv3 architeture, which which you can load into the imageai.Detection.Custom.CustomObjectDetection class. This allows you to train your own model on any set of images that corresponds to any type of objects of interest. The training process generates a JSON file that maps the objects names in your image dataset and the detection anchors, as well as creates lots of models. In choosing the best model for your custom object detection task, an evaluateModel() function has been provided to compute the mAP of your saved models by allowing you to state your desired IoU and Non-maximum Suppression values. Then you can perform custom object detection using the model and the JSON file generated.


- Preparing your custom dataset
- Training on your custom Dataset
- Evaluating your saved detection models' mAP

### Preparing your custom datasets

To train a custom detection model, you need to prepare the images you want to use to train the model. You will prepare the images as follows:

1. Decide the type of object(s) you want to detect and collect about 200 (minimum recommendation) or more picture of each of the object(s)
2. Once you have collected the images, you need to annotate the object(s) in the images. ImageAI uses the Pascal VOC format for image annotation. You can generate this annotation for your images using the easy to use LabelImg image annotation tool, available for Windows, Linux and MacOS systems. Open the link below to install the annotation tool. See: https://github.com/tzutalin/labelImg
3. When you are done annotating your images, annotation XML files will be generated for each image in your dataset. The annotation XML file describes each or all of the objects in the image. For example, if each image your image names are image(1).jpg, image(2).jpg, image(3).jpg till image(z).jpg; the corresponding annotation for each of the images will be image(1).xml, image(2).xml, image(3).xml till image(z).xml.This is to reference or map the objects for detection in json.
4. Once you have the annotations for all your images, create a folder for your dataset (E.g headsets) and in this parent folder, create child folders train and validation
5. In the train folder, create images and annotations sub-folders. Put about 70-80% of your dataset of each object's images in the images folder and put the corresponding annotations for these images in the annotations folder.
6. In the validation folder, create images and annotations sub-folders. Put the rest of your dataset images in the images folder and put the corresponding annotations for these images in the annotations folder.
7. Once you have done this, the structure of your image dataset folder should look like below:
```
>> train    >> images       >> img_1.jpg  (shows Object_1)
            >> images       >> img_2.jpg  (shows Object_2)
            >> images       >> img_3.jpg  (shows Object_1, Object_3 and Object_n)
            >> annotations  >> img_1.xml  (describes Object_1)
            >> annotations  >> img_2.xml  (describes Object_2)
            >> annotations  >> img_3.xml  (describes Object_1, Object_3 and Object_n)

>> validation   >> images       >> img_151.jpg (shows Object_1, Object_3 and Object_n)
                >> images       >> img_152.jpg (shows Object_2)
                >> images       >> img_153.jpg (shows Object_1)
                >> annotations  >> img_151.xml (describes Object_1, Object_3 and Object_n)
                >> annotations  >> img_152.xml (describes Object_2)
                >> annotations  >> img_153.xml (describes Object_1)
```

8. You can train your custom detection model completely from scratch or use transfer learning (recommended for better accuracy) from a pre-trained YOLOv3 model. Also, we have provided a sample annotated Hololens and Headsets (Hololens and Oculus) dataset for you to train with. Download the pre-trained YOLOv3 model and the sample datasets in the link below.

Link to source code of ImageAI by Moses Olfenwa- https://github.com/OlafenwaMoses/ImageAI/releases/tag/essential-v4

### Training on your custom dataset


Before you start training your custom detection model, kindly take note of the following:

1. The default batch_size is 4. If you are training with Google Colab, this will be fine. However, I will advice you use a more powerful GPU than the K80 offered by Colab as the higher your batch_size (8, 16), the better the accuracy of your detection model.
2. If you experience '_TfDeviceCaptureOp' object has no attribute '_set_device_from_string' error in Google Colab, it is due to a bug in Tensorflow. You can solve this by installing Tensorflow GPU 1.13.1.

```
pip3 install tensorflow-gpu==1.13.1
```

Then your training code goes as follows:

```
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
# In the above,when training for detecting multiple objects,
#set object_names_array=["object1", "object2", "object3",..."objectz"]
trainer.trainModel()

```

In the line above, we configured our detection model trainer. The parameters we stated in the function as as below:

- num_objects : this is an array containing the names of the objects in our dataset
- batch_size : this is to state the batch size for the training
- num_experiments : this is to state the number of times the network will train over all the training images, which is also called epochs
- train_from_pretrained_model(optional) : this is to train using transfer learning from a pre-trained YOLOv3 model

trainer.trainModel()

When you start the training, you should see something like this in the console:

```
Using TensorFlow backend.
Generating anchor boxes for training images and annotation...
Average IOU for 9 anchors: 0.78
Anchor Boxes generated.
Detection configuration saved in  hololens/json/detection_config.json
Training on: 	['hololens']
Training with Batch Size:  4
Number of Experiments:  200

Epoch 1/200
 - 733s - loss: 34.8253 - yolo_layer_1_loss: 6.0920 - yolo_layer_2_loss: 11.1064 - yolo_layer_3_loss: 17.6269 - val_loss: 20.5028 - val_yolo_layer_1_loss: 4.0171 - val_yolo_layer_2_loss: 7.5175 - val_yolo_layer_3_loss: 8.9683
Epoch 2/200
 - 648s - loss: 11.1396 - yolo_layer_1_loss: 2.1209 - yolo_layer_2_loss: 4.0063 - yolo_layer_3_loss: 5.0124 - val_loss: 7.6188 - val_yolo_layer_1_loss: 1.8513 - val_yolo_layer_2_loss: 2.2446 - val_yolo_layer_3_loss: 3.5229
Epoch 3/200
 - 674s - loss: 6.4360 - yolo_layer_1_loss: 1.3500 - yolo_layer_2_loss: 2.2343 - yolo_layer_3_loss: 2.8518 - val_loss: 7.2326 - val_yolo_layer_1_loss: 1.8762 - val_yolo_layer_2_loss: 2.3802 - val_yolo_layer_3_loss: 2.9762
Epoch 4/200
 - 634s - loss: 5.3801 - yolo_layer_1_loss: 1.0323 - yolo_layer_2_loss: 1.7854 - yolo_layer_3_loss: 2.5624 - val_loss: 6.3730 - val_yolo_layer_1_loss: 1.4272 - val_yolo_layer_2_loss: 2.0534 - val_yolo_layer_3_loss: 2.8924
Epoch 5/200
 - 645s - loss: 5.2569 - yolo_layer_1_loss: 0.9953 - yolo_layer_2_loss: 1.8611 - yolo_layer_3_loss: 2.4005 - val_loss: 6.0458 - val_yolo_layer_1_loss: 1.7037 - val_yolo_layer_2_loss: 1.9754 - val_yolo_layer_3_loss: 2.3667
Epoch 6/200
 - 655s - loss: 4.7582 - yolo_layer_1_loss: 0.9959 - yolo_layer_2_loss: 1.5986 - yolo_layer_3_loss: 2.1637 - val_loss: 5.8313 - val_yolo_layer_1_loss: 1.1880 - val_yolo_layer_2_loss: 1.9962 - val_yolo_layer_3_loss: 2.6471
Epoch 7/200
```


```
Using TensorFlow backend.
Generating anchor boxes for training images and annotation...
Average IOU for 9 anchors: 0.78
Anchor Boxes generated.
Detection configuration saved in  hololens/json/detection_config.json
Training on: 	['hololens']
Training with Batch Size:  4
Number of Experiments:  200

```
The above details signifies the following:

1. ImageAI autogenerates the best match detection anchor boxes for your image dataset.
2. The anchor boxes and the object names mapping are saved in json/detection_config.json path of in the image dataset folder. Please note that for every new training you start, a new detection_config.json file is generated and is only compatible with the model saved during that training.

```
Epoch 1/200
 - 733s - loss: 34.8253 - yolo_layer_1_loss: 6.0920 - yolo_layer_2_loss: 11.1064 - yolo_layer_3_loss: 17.6269 - val_loss: 20.5028 - val_yolo_layer_1_loss: 4.0171 - val_yolo_layer_2_loss: 7.5175 - val_yolo_layer_3_loss: 8.9683
Epoch 2/200
 - 648s - loss: 11.1396 - yolo_layer_1_loss: 2.1209 - yolo_layer_2_loss: 4.0063 - yolo_layer_3_loss: 5.0124 - val_loss: 7.6188 - val_yolo_layer_1_loss: 1.8513 - val_yolo_layer_2_loss: 2.2446 - val_yolo_layer_3_loss: 3.5229
Epoch 3/200
 - 674s - loss: 6.4360 - yolo_layer_1_loss: 1.3500 - yolo_layer_2_loss: 2.2343 - yolo_layer_3_loss: 2.8518 - val_loss: 7.2326 - val_yolo_layer_1_loss: 1.8762 - val_yolo_layer_2_loss: 2.3802 - val_yolo_layer_3_loss: 2.9762
Epoch 4/200
 - 634s - loss: 5.3801 - yolo_layer_1_loss: 1.0323 - yolo_layer_2_loss: 1.7854 - yolo_layer_3_loss: 2.5624 - val_loss: 6.3730 - val_yolo_layer_1_loss: 1.4272 - val_yolo_layer_2_loss: 2.0534 - val_yolo_layer_3_loss: 2.8924
Epoch 5/200
 - 645s - loss: 5.2569 - yolo_layer_1_loss: 0.9953 - yolo_layer_2_loss: 1.8611 - yolo_layer_3_loss: 2.4005 - val_loss: 6.0458 - val_yolo_layer_1_loss: 1.7037 - val_yolo_layer_2_loss: 1.9754 - val_yolo_layer_3_loss: 2.3667
Epoch 6/200
 - 655s - loss: 4.7582 - yolo_layer_1_loss: 0.9959 - yolo_layer_2_loss: 1.5986 - yolo_layer_3_loss: 2.1637 - val_loss: 5.8313 - val_yolo_layer_1_loss: 1.1880 - val_yolo_layer_2_loss: 1.9962 - val_yolo_layer_3_loss: 2.6471
Epoch 7/200
```

3. The above signifies the progress of the training.
4. For each experiment (Epoch), the general total validation loss (E.g - loss: 4.7582) is reported.
5. For each drop in the loss after an experiment, a model is saved in the hololens/models folder. The lower the loss, the better the model.

Once you are done training, you can visit the link below for performing object detection with your custom detection model and detection_config.json file.

### Evaluating your saved detection models' mAP

After training on your custom dataset, you can evaluate the mAP of your saved models by specifying your desired IoU and Non-maximum suppression values. See details as below:

1. Single Model Evaluation: To evaluate a single model, simply use the example code below with the path to your dataset directory, the model file and the detection_config.json file saved during the training. In the example, we used an object_threshold of 0.3 ( percentage_score >= 30% ), IoU of 0.5 and Non-maximum suppression value of 0.5.

```
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
metrics = trainer.evaluateModel(model_path="detection_model-ex-60--loss-2.76.h5", json_path="detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
```


Consider that trainer.evaluateModel method will show the metrics on standard output as shown below, but also returns a list of dicts containing all the information that is displayed.
```
Sample Result:

Model File:  hololens_detection_model-ex-09--loss-4.01.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9613
mAP: 0.9613
===============================

Let's see how those metrics looks like:

[{
    'average_precision': {'hololens': 0.9613334437735249},
    'map': 0.9613334437735249,
    'model_file': 'hololens_detection_model-ex-09--loss-4.01.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
}]

```

2. Multi Model Evaluation: To evaluate all your saved models, simply parse in the path to the folder containing the models as the model_path as seen in the example below:

```
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
metrics = trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
```

Sample Result:

```
Model File:  hololens/models/detection_model-ex-07--loss-4.42.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9231
mAP: 0.9231
===============================
Model File:  hololens/models/detection_model-ex-10--loss-3.95.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9725
mAP: 0.9725
===============================
Model File:  hololens/models/detection_model-ex-05--loss-5.26.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9204
mAP: 0.9204
===============================
Model File:  hololens/models/detection_model-ex-03--loss-6.44.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.8120
mAP: 0.8120
===============================
Model File:  hololens/models/detection_model-ex-18--loss-2.96.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9431
mAP: 0.9431
===============================
Model File:  hololens/models/detection_model-ex-17--loss-3.10.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9404
mAP: 0.9404
===============================
Model File:  hololens/models/detection_model-ex-08--loss-4.16.h5 
Using IoU :  0.5
Using Object Threshold :  0.3
Using Non-Maximum Suppression :  0.5
hololens: 0.9725
mAP: 0.9725
===============================
Let's see how those metrics looks like:

[{
    'average_precision': {'hololens': 0.9231334437735249},
    'map': 0.9231334437735249,
    'model_file': 'hololens/models/detection_model-ex-07--loss-4.42.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
},
{
    'average_precision': {'hololens': 0.9725334437735249},
    'map': 0.97251334437735249,
    'model_file': 'hololens/models/detection_model-ex-10--loss-3.95.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
},
{
    'average_precision': {'hololens': 0.92041334437735249},
    'map': 0.92041334437735249,
    'model_file': 'hololens/models/detection_model-ex-05--loss-5.26.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
},
{
    'average_precision': {'hololens': 0.81201334437735249},
    'map': 0.81201334437735249,
    'model_file': 'hololens/models/detection_model-ex-03--loss-6.44.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
},
{
    'average_precision': {'hololens': 0.94311334437735249},
    'map': 0.94311334437735249,
    'model_file': 'hololens/models/detection_model-ex-18--loss-2.96.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
},
{
    'average_precision': {'hololens': 0.94041334437735249},
    'map': 0.94041334437735249,
    'model_file': 'hololens/models/detection_model-ex-17--loss-3.10.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
},
{
    'average_precision': {'hololens': 0.97251334437735249},
    'map': 0.97251334437735249,
    'model_file': 'hololens/models/detection_model-ex-08--loss-4.16.h5',
    'using_iou': 0.5,
    'using_non_maximum_suppression': 0.5,
    'using_object_threshold': 0.3
}
]
```







# ImageAI : Custom Object Detection


- Custom Object Detection
- Object Detection, Extraction and Fine-tune
- Hiding/Showing Object Name and Probability
- Image Input & Output Types


ImageAI provides very convenient and powerful methods to perform object detection on images and extract each object from the image using your own custom YOLOv3 model and the corresponding detection_config.json generated during the training. To test the custom object detection, you can download a sample custom model we have trained to detect the Hololens headset and its detection_config.json file via the links below:
```
- https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/hololens-ex-60--loss-2.76.h5
- https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/detection_config.json
```
Once you download the custom object detection model file, you should copy the model file to the your project folder where your .py files will be. Then create a python file and give it a name; an example is FirstCustomDetection.py. Then write the code below into the python file:


***FirstCustomDetection.py***
 
 ```
 from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("hololens-ex-60--loss-2.76.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
```


***Object Detection, Extraction and Fine-tune***

```
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("hololens-ex-60--loss-2.76.h5")
detector.setJsonPath("detection_config.json") 
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="holo2.jpg", output_image_path="holo2-detected.jpg", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")
    
```

***Hiding/Showing Object Name and Probability***


```
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "holo2.jpg"), output_image_path=os.path.join(execution_path , "holo2_nodetails.jpg"), minimum_percentage_probability=30, display_percentage_probability=False, display_object_name=False)

```


***Image Input & Output Types***


ImageAI custom object detection supports 2 input types of inputs which are file path to image file(default) and numpy array of an image as well as 2 types of output which are image file(default) and numpy **array **. This means you can now perform object detection in production applications such as on a web server and system that returns file in any of the above stated formats. To perform object detection with numpy array input, you just need to state the input type in the .detectObjectsFromImage() function. See example below.


```
detections = detector.detectObjectsFromImage(input_type="array", input_image=image_array , output_image_path=os.path.join(execution_path , "holo2-detected.jpg")) # For numpy array input type
```

To perform object detection with numpy array output you just need to state the output type in the .detectObjectsFromImage() function. See example below.

```
detected_image_array, detections = detector.detectObjectsFromImage(output_type="array", input_image="holo2.jpg" ) # For numpy array output type
```

# ImageAI : Custom Video Object Detection, Tracking and Analysis

- First Custom Video Object Detection
- Camera / Live Stream Video Detection
- Video Analysis
- Hiding/Showing Object Name and Probability
- Frame Detection Intervals
- Video Detection Timeout (NEW)


ImageAI provides convenient, flexible and powerful methods to perform object detection on videos using your own custom YOLOv3 model and the corresponding detection_config.json generated during the training. 


Because video object detection is a compute intensive tasks, we advise you perform this experiment using a computer with a NVIDIA GPU and the GPU version of Tensorflow installed. Performing Video Object Detection CPU will be slower than using an NVIDIA GPU powered computer. You can use Google Colab for this experiment as it has an NVIDIA K80 GPU available for free.

***FirstCustomVideoObjectDetection.py***

```

from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)
					  
					  
```

***Camera / Live Stream Video Detection***

ImageAI now allows live-video detection with support for camera inputs. Using OpenCV's VideoCapture() function, you can load live-video streams from a device camera, cameras connected by cable or IP cameras, and parse it into ImageAI's detectObjectsFromVideo() function. All features that are supported for detecting objects in a video file is also available for detecting objects in a camera's live-video feed. Find below an example of detecting live-video feed from the device camera.

```
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,
                                          minimum_percentage_probability=40,
                                          log_progress=True)


```



The difference in the code above and the code for the detection of a video file is that we defined an OpenCV VideoCapture instance and loaded the default device camera into it. Then we parsed the camera we defined into the parameter camera_input which replaces the input_file_path that is used for video file.


***Video Analysis***
ImageAI now provide commercial-grade video analysis in the Custom Video Object Detection class, for both video file inputs and camera inputs. This feature allows developers to obtain deep insights into any video processed with ImageAI. This insights can be visualized in real-time, stored in a NoSQL database for future review or analysis.

For video analysis, the detectObjectsFromVideo() now allows you to state your own defined functions which will be executed for every frame, seconds and/or minute of the video detected as well as a state a function that will be executed at the end of a video detection. Once this functions are stated, they will receive raw but comprehensive analytical data on the index of the frame/second/minute, objects detected (name, percentage_probability and box_points), number of instances of each unique object detected and average number of occurrence of each unique object detected over a second/minute and entire video.

To obtain the video analysis, all you need to do is specify a function, state the corresponding parameters it will be receiving and parse the function name into the per_frame_function, per_second_function, per_minute_function and video_complete_function parameters in the detection function. Find below examples of video analysis functions.

```
def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20, per_second_function=forSeconds, per_frame_function = forFrame, per_minute_function= forMinute,
                                          minimum_percentage_probability=40,
                                          log_progress=True)
```


ImageAI also allows you to obtain complete analysis of the entire video processed. All you need is to define a function like the forSecond or forMinute function and set the video_complete_function parameter into your .detectObjectsFromVideo() function. The same values for the per_second-function and per_minute_function will be returned. The difference is that no index will be returned and the other 3 values will be returned, and the 3 values will cover all frames in the video. Below is a sample function:
```
def forFull(output_arrays, count_arrays, average_output_count):
    #Perform action on the 3 parameters returned into the function


video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          video_complete_function=forFull,
                                          minimum_percentage_probability=40,
                                          log_progress=True)
```

FINAL NOTE ON VIDEO ANALYSIS : 
ImageAI allows you to obtain the detected video frame as a Numpy array at each frame, second and minute function. All you need to do is specify one more parameter in your function and set return_detected_frame=True in your detectObjectsFromVideo() function. Once this is set, the extra parameter you sepecified in your function will be the Numpy array of the detected frame. See a sample below:

```
def forFrame(frame_number, output_array, output_count, detected_frame):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
	print("Returned Objects is : ", type(detected_frame))
    print("------------END OF A FRAME --------------")


video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          per_frame_function=forFrame,
                                          minimum_percentage_probability=40,
                                          log_progress=True, return_detected_frame=True)
```
***Frame Detection Intervals***

The above video objects detection task are optimized for frame-real-time object detections that ensures that objects in every frame of the video is detected. ImageAI provides you the option to adjust the video frame detections which can speed up your video detection process. When calling the .detectObjectsFromVideo(), you can specify at which frame interval detections should be made. By setting the frame_detection_interval parameter to be equal to 5 or 20, that means the object detections in the video will be updated after 5 frames or 20 frames. If your output video frames_per_second is set to 20, that means the object detections in the video will be updated once in every quarter of a second or every second. This is useful in case scenarios where the available compute is less powerful and speeds of moving objects are low. This ensures you can have objects detected as second-real-time , half-a-second-real-time or whichever way suits your needs.

***Custom Video Detection Timeout***

ImageAI now allows you to set a timeout in seconds for detection of objects in videos or camera live feed. To set a timeout for your video detection code, all you need to do is specify the detection_timeout parameter in the detectObjectsFromVideo() function to the number of desired seconds. In the example code below, we set detection_timeout to 120 seconds (2 minutes).

```

from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(0)

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "holo1-detected3"),
                                          frames_per_second=20,  minimum_percentage_probability=40,
                                          detection_timeout=120)

```



We have trained some logos on Nike and Adidas through ImageAI for detection.If any queries,please contact below-

- Shilpa S J
Email: shilpatc25@gmail.com
Facebook: https://www.facebook.com/shilpasj.25


***Citation***

We can cite ImageAI in your projects and research papers via the BibTeX entry below.

@misc {ImageAI,
    author = "Moses and John Olafenwa",
    title  = "ImageAI, an open source python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities",
    url    = "https://github.com/OlafenwaMoses/ImageAI",
    month  = "mar",
    year   = "2018--"
}

***References***

1. https://github.com/OlafenwaMoses/ImageAI
2. Somshubra Majumdar, DenseNet Implementation of the paper, Densely Connected Convolutional Networks in Keras https://github.com/titu1994/DenseNet
3. Broad Institute of MIT and Harvard, Keras package for deep residual networks https://github.com/broadinstitute/keras-resnet
4. Experiencor, Training and Detecting Objects with YOLO3 https://github.com/experiencor/keras-yolo3
5. https://www.tensorflow.org/guide
6. https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
7. O Russakovsky et al, ImageNet Large Scale Visual Recognition Challenge https://arxiv.org/abs/1409.0575
8. Fizyr, Keras implementation of RetinaNet object detection https://github.com/fizyr/keras-retinanet
9. Forrest N. et al, SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size https://arxiv.org/abs/1602.07360
10. Francois Chollet, Keras code and weights files for popular deeplearning models https://github.com/fchollet/deep-learning-models
11. Kaiming H. et al, Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385
12. Szegedy. et al, Rethinking the Inception Architecture for Computer Vision https://arxiv.org/abs/1512.00567
13. Gao. et al, Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993
14. TY Lin et al, Microsoft COCO: Common Objects in Context https://arxiv.org/abs/1405.0312
15. Moses & John Olafenwa, A collection of images of identifiable professionals. https://github.com/OlafenwaMoses/IdenProf
16. Joseph Redmon and Ali Farhadi, YOLOv3: An Incremental Improvement. https://arxiv.org/abs/1804.02767
17. Tsung-Yi. et al, Focal Loss for Dense Object Detection https://arxiv.org/abs/1708.02002
18. https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85


