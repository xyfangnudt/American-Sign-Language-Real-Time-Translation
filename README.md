Real-time American Sign Language (ASL) Fingerspelling Translation
================================================================

Introduction
------------
American Sign Language (ASL) is the predominant sign language for the Deaf community 
in the United States. Because most signs involve movement and facial expressions, 
this project aims to only recognize ASL fingerspelling.

Below is an image of signs that this program will recognize.

![ASL Signs](signs.png)

Requirements
------------

* Numpy (1.13.0)
* Scikit-Learn (0.18.1)
* OpenCV (3.0)
* Scipy (0.19.0)
* Python3

Installation
-------------
* `git clone https://github.com/GarrettBeatty/American-Sign-Language-Real-Time-Translation.git`
* `pip3 install -r requirements.txt` 

Running
-------
* ` python3 webcam.py`

How it Works                                            
-------------------

* Extract features from the training images using the [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) algorithm.
* Create a dictionary of visual words using the KMeans algorithm.
* Create a Bag of Words (BoW) model for each image.
* Feed the BoW model into a machine learning algorithm. (This program using a Support Vector Machine.)
    
Results
-------------

Author
-------------
* Garrett Beatty

License
-------------
See [License](#LICENSE.md)

More Info
---------
[https://en.wikipedia.org/wiki/American_Sign_Language](https://en.wikipedia.org/wiki/American_Sign_Language)