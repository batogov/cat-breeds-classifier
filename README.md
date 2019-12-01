# Cat Breeds Classifier

Read the [Russian version](./README_RU.md) of the document 🇷🇺

## Dependencies

* **Python 3**;
* **Tensorflow** and additional stuff for it.

## Theory

At this moment, the problem of image classification is best solved by **convolutional neural networks** (CNN). The main idea of convolutional neural networks lies in the alternation of convolutional layers and subsampling layers.

Работа свёрточной нейронной сети обычно интерпретируется как переход от конкретных особенностей изображения к более абстрактным деталям, и далее к ещё более абстрактным деталям вплоть до выделения понятий высокого уровня. При этом сеть самонастраивается и вырабатывает сама необходимую иерархию абстрактных признаков (последовательности карт признаков), фильтруя маловажные детали и выделяя существенное.

CNN (как и любая нейронная сеть) для обучения на более или менее большой выборке данных требует больших затрат мощности (CPU или GPU) и времени. Чтобы избежать этого, в данной работе была использована предобученная свёрточная нейронная сеть **Inception** от **Google**.

Процесс «дообучения» нейронной сети называется **transfer learning**. По факту, мы берём полностью готовую, обученную на огромной количестве изображений (например, на базе ImageNet) модель и переобучаем (с использованием уже наших изображений) последний её слой.

## About dataset

In this work we used images of cats of the following breeds: Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx. The images were taken from [here](http://www.robots.ox.ac.uk/~vgg/data/pets/).

Since this neural network is able to work only with images in jpeg format, then it was necessary to delete all images of other formats. For finding and deleting such files (e.g. gif files that were renamed to jpeg) is used *check_file_extension.py* script.

## Обучение

To start the process of transfer learning the network is used bash script *train.sh*. В этом файле необходимо указать путь до файла *retrain.py* из *examples/image_retraining* репозитория библиотеки tensorflow, пути до изображений, служебных файлов и количество итераций обучения.

## Prediction

For prediction is used *predict.py*. You need to pass the path to the image as an argument at startup.
