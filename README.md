# Semantic Segmentation of Agricultural Fields

Class project for CS 4774 - Machine Learning at the University of Virginia with a focus on utilizing machine learning techniques to benefit the Commonwealth of Virginia.

# Motivation

The goal of our project was to use computer vision techniques to identify agricultural fields close in proximity to the Chesapeake Bay Watershed and its estuaries. We researched and developed a model capable of generating semantic segmentation boundaries of agricultural fields from satellite images. The predictions generated from this model could then be used to identify farms at a higher risk of polluting the Bay.

# Use Instructions

To run the project, it is recommended to use [Google Colaboratory](https://colab.research.google.com/) because it has the necessary python modules pre-installed, as well as a GPU for faster training and inference speeds.

If you have a GPU with >8 GB of memory available (and a functioning CUDA installation), you can download the necessary packages using `pip install -r requirements.txt`.

# Credits

This project was inspired by the work of Christoph Rieke, which can be found [here](https://github.com/chrieke/InstanceSegmentation_Sentinel2). His data preprocessing techniques were used to obtain the images found in our data directory.
