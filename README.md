# BERT Fine-Tuned for Binary Sentiment Classification on Sarcasm Detection
## 1) Introduction
- This project focuses on fine-tuning Google's Bidirectional Encoder Representations from Transformers (BERT) to classify sarcastic news headlines. The goal is to familiarize with BERT's mechanisms, particularly modifying the default maximum token length for better performance on varying text lengths. The model was trained and tested on Kaggle, which has recently become a hub for TensorFlow models due to TensorFlow Hub's integration.

## 2) Model Performance
- The fine-tuned model was trained on kaggle with Nvidia Tesla P100 and batch_size=16 and shows promising results in classifying sarcastic versus non-sarcastic news headlines:

  1) Training AUC Score: 0.9868
  2) Validation AUC Score: 0.9825
  3) Maximum F1 Score: 0.935 at a prediction threshold of 0.17.

###### Note: The decision boundary was tuned to optimize the F1 score. Depending on the specific use-case, the threshold can be adjusted to prioritize reducing false positives or negatives.

## 3) Components
1. BERT Encoder
Version:  [bert-en-uncased-l-12-h-768-a-12 version 2](https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/bert-en-uncased-l-12-h-768-a-12/versions/2)
2. BERT Preprocessor
Version: [en-uncased-preprocess version 3](https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-preprocess/versions/3)
3. Dataset
Source: [News Headlines Dataset For Sarcasm Detection v2](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

## 4) Code Structure
The project is organized into four major sections:

- Input Pipelines:
  1) This segment of the pipeline takes the sanitized dataset and converts it into a TensorFlow dataset format. It's configured to prefetch data using the CPU, thereby optimizing the pipeline's efficiency and runtime performance, particularly for model training on a GPU.
  2) This process is done on both training and validation data.
- BertPreprocessor Pipeline:
  1) TThis class is designed to accept a batch of raw text strings and a specified sequence length (seq_length) as inputs. It formats the text into a structure conducive for BERT processing, delineating 'input_word_ids', 'input_mask', and 'input_type_ids'. The sequence length parameter allows for modification of BERT's default maximum token count without hassles.
  2) Integral to the pipeline, this preprocessor class doesn't require separate initialization for model execution as it's already incorporated within the subsequent classifier class. However, it retains the flexibility to be independently initialized for experimental or testing purposes outside of the model's main flow.
- BertClassifier Pipeline:
  1) This class serves as the core of the machine learning pipeline. It is responsible for initializing the classifier, integrating the BertPreprocessor and the BERT encoder, along with the defined sequence length (seq_length). This configuration ensures that the classifier is suitably prepared with the necessary components and parameters for effective model training. Once the BertClassifier is initialized, it stands ready to proceed with the training process.

## 5) Getting Started
To use this model for your purposes or to replicate the results:

Follow my kaggle notebook [BERTed_sarcastic_news_headlines](https://www.kaggle.com/code/omarxz/berted-sarcastic-news-headlines) and the rest is self explanatory.



### Contact
For further inquiries or potential collaborations, feel free to contact [oramadan@hawaii.edu](oramadan@hawaii.edu).
