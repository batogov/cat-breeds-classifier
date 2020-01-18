# Cat Breeds Classifier

Read the [Russian version](./README_RU.md) of the document 🇷🇺

## Dependencies

* **Python 3**;
* **Tensorflow** and additional stuff for it.

## Theory

At this moment, the problem of image classification is best solved by **convolutional neural networks** (CNN). The main idea of convolutional neural networks lies in the alternation of convolutional layers and subsampling layers.

The work of a convolutional neural network is usually interpreted as a transition from specific features of an image to more abstract details, and then to even more abstract details, up to the allocation of high-level concepts. At the same time, the network adjusts itself and develops the necessary hierarchy of abstract features, filtering unimportant details and highlighting the essential ones.

CNN (like any neural network) requires a lot of CPU or GPU resources and time to train on a large sample of data. To avoid this, I used the pre-trained convolutional neural network **Inception** from **Google**.

The process of additional neural network training is called **transfer learning**. In fact, I take a completely ready-made model trained on a huge number of images (for example, based on ImageNet) and retrain (using our images) the last layer of it.

## About dataset

In this work we used images of cats of the following breeds: Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx. The images were taken from [here](http://www.robots.ox.ac.uk/~vgg/data/pets/).

Since this neural network is able to work only with images in jpeg format, then it was necessary to delete all images of other formats. For finding and deleting such files (e.g. gif files that were renamed to jpeg) is used *check_file_extension.py* script.

## Transfer learning

To start the process of transfer learning the network is used bash script *train.sh*. In this file, you must specify the path to the file *examples/image_retraining/retrain.py* form the tensorflow repository, the path to images and service files, and the number of training iterations.

## Prediction

For prediction is used *predict.py*. You need to pass the path to the image as an argument at startup.
