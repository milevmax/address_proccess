# Similarity Address Pipeline

This project is a Python-based pipeline for comparing and calculating similarity between address pairs. The project uses various techniques, including fuzzy string matching and a neural network model, to generate similarity scores and classify address pairs.

## Project Overview
The Similarity Address Pipeline is designed to process and compare address pairs using fuzzy matching algorithms and a neural network model. It allows for preprocessing of address data, calculation of fuzzy similarity scores, and classification of address pairs based on similarity thresholds.

## Features
  
- The pipeline includes a `method` parameter that can be set to either `'nn'` or `'combo'` to specify how data should be processed and labeled:  
  - **combo**
    - using simple fuzzy similarity score higher than 0.85 is marked as a match (`1`), while a score lower than 0.25 is marked as not a match (`0`).
    - using nn similarity score for all other samples (0.25 < fuzzy < 0.85)  
     This method is fast and straightforward but may sometimes yield false positives when there are small but significant differences between addresses. For example:   
     *місто київ, оболонський р-н, оболонський проспект 19, кв 122*  
     *місто київ, оболонський р-н, оболонський проспект 12, кв 77*  
     fuzzy_score: **0.95**  , nn_score: 0.015254
  - **nn**
    - using TensorFlow neural network model "nn" similarity score for all samples without accounting fuzzy_score
- Customizable similarity thresholds.
- Easy-to-extend code structure for adding new comparison methods.

### Runtime  
Device: GeForce RTX 3050 Mobile, 12th Gen Intel® Core™ i7-12650H × 16   
10000/10000 [==============================] - 62s 6ms/step - [tf model recognition 10000 batches with batchsize = 32 (320000 address pairs)]  
Runtime : 81.45 seconds - full pipeline time

### Launching
[SimilarityAddressPipeline.py](SimilarityAddressPipeline.py)