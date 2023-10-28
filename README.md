# Muse

![](https://img.shields.io/badge/-Python%203.11-ffd343?logo=python)
![](https://img.shields.io/badge/-PyTorch%202.0-lightgray?logo=pytorch)
![](https://img.shields.io/badge/-Scikit--learn%201.3.0-gray?logo=scikitlearn)
![](https://img.shields.io/badge/-Flask%202.3.2-333?logo=flask)

Muse is an artificial intelligence research project exploring content-based music recommendation. \
It investigates two recommenders that make use of some given songs to try and recommend similar yet diverse music.

This research was carried out in the context of an MSc in Artificial Intelligence. 

## Content
This repo features two files: \
A jupyter notebook documenting the research process and a python file presenting the API to run the two recommenders.

```Final_Project.ipynb``` \
This file is split into 3 main sections: Processing Jukebox representations, training the Key/Emotion MIR models, and creating the recommenders. Since it's a notebook file, most outputs cells have been kept and all the code is nicely split into cells for better understanding.

```app.py``` \
This is the API file using flask 2.3.2 to receive recommendation requests and send back the recommended song ids.
It loads the clustered recommender models processed in the Final_Project notebook and runs on CPU to process the recommendations.

## Credits
Part of this work was possible thanks to the research carried out by Rodrigo Castellon, Chris Donahue, and Percy Liang on [Codified Audio Language Modelling for MIR tasks](https://arxiv.org/abs/2107.05677).

A huge thanks to [Mathieu Harmant](https://harmant-mathieu.fr/) for his help implementing the [Muse website](https://muse.augustindirand.com/)  communicating with the API and allowing users to test and evaluate the recommenders.
