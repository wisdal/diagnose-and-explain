# Automated Chest Radiology Reporting from X-ray using a CNN-LSTM model and a multi-level Attention Mechanism

This repository contains code for the paper: "Diagnose and Explain: A Multi-Level Attention Model for Automated Radiology Reporting from Chest X-Ray"

Our model takes a Chest X-ray image as input and generates a complete radiology report using Natural Langugage. The report contains 2 sections:
* **Findings:** observations regarding each part of the chest examined. Generally a paragraph with 6+ sentences.
* **Impression:** generally a one-sentence diagnostic based on findings reported. Can contain multiple sentences.

## Dataset

We trained our model on the Indiana University [Chest X-Ray collection](https://openi.nlm.nih.gov/faq.php). The dataset
comes with **3955** chest radiology reports from various hospital systems and 7470 associated chest x-rays 
(most reports are associated with 2 or more images representing frontal and lateral views).

## Model architecture

Details of our model and proposed approach will be made available in a paper preprint soon to be released.

Model Highlights:
*
*
## Source structure

The code comes with 3 files: 
* model: class declarations and model training code
* prepare_dataset: dataset download and data preprocessing
* main: main file to be run on Cloud TPU

## Training on Cloud TPU

## Predicting

## Results

## Acknowledgements

Parts of these experiments were possible thanks to the TensorFlow Research Cloud Program which offers researchers free TPUs for training and running models on Google Cloud for a limited period of time.
