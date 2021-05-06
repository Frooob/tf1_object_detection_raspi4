# Embedded Object Detection with the Tensorflow 1 Object Detection API using the raspi 4

In this repo, you find a documentation and an example on how to use the 
TF1 Object Detection API to train your own custom quantized object dtection model. 
After the model is trained, it will be converted to a tflite model and then
compiled for the google edge tpu accelerator. You can find a live example of 
the project on [YouTube](https://youtu.be/eHBGWim-vzw).

Special thanks to the [TensorFlow Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/)
and [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)
that provided a solid base for the used code. 

## Preparing your data

Start of by taking a lot of pictures of your interested objects with heterogenous 
backgrounds. You can afterwards use [LabelImg](https://github.com/tzutalin/labelImg)
to label the objects in your pictures. 
To be able to train tensorflow with your images, you need to convvert them into 
the `tfrecord` format. Follow the instructions in the [TensorFlow Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/) 
to split your data into a training and a test dataset. 

## Training your model

In `train_and_eval_mode.ipynb`, you can find the whole process of installing dependencies, 
fetching your tfrecords from github (you can also use your local tfrecords), 
training and evaluating the model.


## Compiling the model for the edge tpu

To run the model on the raspi 4 it needs to be in the `.tfite` format. 
If you additionally want to speed up the inference with the edgetpu, the model 
needs to be specifically compiled for that hardware. 
For this purpose, I wrote 
a Dockerfile in `./edgetpu_compiler` that you can use to compile your model for the edge tpu. 
For convenience, I wrote a script `./convert_tflite.py` that takes as an input the
model that you get from your training, and converts it into a edge tpu compatible `.tflite` model. Just make sure you built the Docker file before. 


## Using the model on the raspi 4

I modified a webcam-detection script from [EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)
that starts up the webcam on a raspberry pi, uses a given model and drives LEDs and a little speaker. 
You can find the modified script in `./detect_webcam.py`. 
A live example of the raspi detecting the controllers of the Nintendo switch can be seen on [YouTube](https://youtu.be/eHBGWim-vzw)
