# ETMS@IITKGP at SemEval-2022 Task 10: Structured Sentiment Analysis Using A Generative Approach 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B5IQ4WUA2XQ77HZv_3EyZG-mArTSWm3n?usp=sharing)

This code in google colab is part of the paper published at SemEval-2022 for the Task 10 - Structured Sentiment Analysis. Please follow the code line by line with the section comments to generate the final output files for the task. 

The inspiration for the model is taken from the ACL 2021 paper, [A Unified Generative Framework for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2106.04300).

## Repository Structure
- BARTABSA/peng : The model for the SSA Task, BARTABSA/peng/train.py consists of the main code which runs to generate the final output files for the datasets specified in the colab  file
- BARTABSA/final_data :  This folder consists of all the dataset files, which consists of both the datasets provided in the task and the processed datasets (processing steps are mentioned in the paper). These files are used to fit to the architecture of the code that we have used. Refer to the colab file for further details.
- BARTABSA: Please refer to the README inside this folder for exact details

## Relevant Links
- [CodaLab](https://competitions.codalab.org/competitions/33556), the competition was hosted
- [Task Github Repository](https://github.com/jerbarnes/semeval22_structured_sentiment), the initial datasets and evaluation scripts were provided
