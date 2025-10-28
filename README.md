                                         # Abhijeet Singh Pawar – Projects Portfolio

Welcome to the project repository of **Abhijeet Singh Pawar**. This collection highlights practical implementations in machine learning, data science, and computer vision, designed to showcase expertise in data analytics, algorithm development, and model deployment using Python and open-source libraries.

---

## Projects

### 1. Automatic License Plate Detection (ALPD)
https://github.com/aspabhi31/PortfolioProjects/blob/main/Automatic%20License%20Plate%20Detection(ALPD).ipynb
https://github.com/aspabhi31/PortfolioProjects/blob/main/Automatic%20License%20Plate%20Detection(ALPD).pdf


**Tech Stack:** Python, OpenCV, NumPy, Matplotlib  
**Overview:**  
Developed an image processing pipeline capable of detecting vehicle license plates under varying lighting and angle conditions.
- Used Gaussian blurring for noise reduction.
- Applied histogram equalization for contrast enhancement.
- Employed adaptive thresholding to handle uneven illumination scenarios.
- Implemented custom contour detection and filtering using geometric criteria (area, aspect ratio, shape).
- Optimized detection accuracy through parameter tuning (Gaussian kernel size, threshold block size, contour area).

---

### 2. Combined Cycle Power Plant Energy Output Prediction
https://github.com/aspabhi31/PortfolioProjects/tree/main/5.%20Project-Gradient%20Descent


**Tech Stack:** Python, NumPy, scikit-learn  
**Overview:**  
Built a custom linear regression model from scratch to predict net hourly electrical energy output for power plant operations.
- Created a custom gradient descent optimizer to minimize mean squared error.
- Modeled relationships between ambient conditions (temperature, pressure, humidity, exhaust vacuum) and energy output.
- Applied feature scaling (StandardScaler) for stable convergence.
- Generated reproducible predictions and provided test set outputs as CSV.

---

### 3. Titanic Survival Prediction using Logistic Regression
https://github.com/aspabhi31/PortfolioProjects/tree/main/7.%20Project-Logistic%20Regression/Logistic%20Regression-Titanic%20Dataset


**Tech Stack:** Python, Pandas, NumPy, scikit-learn  
**Overview:**  
Trained a logistic regression model to predict Titanic passenger survival.
- Performed robust data preprocessing: handling missing values, encoding categorical variables, and extracting informative features (e.g., passenger titles).
- Converted cabin presence to binary indicators, dropped irrelevant fields.
- Tuned logistic regression model (using 'saga' solver, with high max iterations and low tolerance).
- Explored ensemble models like Random Forest for comparison.
- Submitted predictions as CSV, with comprehensive feature engineering.

### 4. Decision Tree Classifier from Scratch
https://github.com/aspabhi31/PortfolioProjects/blob/main/Decision%20Tree%20from%20scratch.ipynb

A Python implementation of a decision tree classifier built from scratch, without relying on machine learning libraries for the core algorithm. The classifier is tested on the Breast Cancer Wisconsin dataset, achieving 92.98% accuracy.

#### Project Overview
This project demonstrates a complete implementation of a decision tree classifier, including:
- Entropy and information gain calculations for optimal splits.
- Recursive tree construction with random feature selection.
- Prediction logic for classifying new samples.
- Testing on the Breast Cancer Wisconsin dataset from scikit-learn.

#### Features
- Implements a `Node` class for tree structure (internal and leaf nodes).
- Supports random feature selection to mimic random forest behavior.
- Uses NumPy for efficient array operations and `Counter` for majority voting.
- Configurable parameters: `min_samples_split`, `max_depth`, `n_features`.

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Decision-Tree-from-Scratch.git

### 5. Text Classification Project
https://github.com/aspabhi31/PortfolioProjects/blob/main/Project-%20Text%20Classification.ipynb

A Python implementation of a decision tree classifier built from scratch, without relying on machine learning libraries for the core algorithm. The classifier is tested on the Breast Cancer Wisconsin dataset, achieving 92.98% accuracy.

#### Project Overview

Dataset: 20 Newsgroups (~11,314 documents across 20 categories)

Task: Multi-class text classification

Features: Bag-of-words with top 3,000 frequent words

Models: Custom Naive Bayes classifier with Laplace smoothing
Scikit-learn's MultinomialNB for comparison

Performance: Achieves ~88% accuracy with 3,000 features

Computation: Takes 10-12 minutes to run due to heavy matrix operations

#### Results

Accuracy: ~88% for both custom and scikit-learn classifiers with 3,000 features.

Confusion Matrix: Shows class-wise performance (Cell 25).

Classification Report: Details precision, recall, and F1-score per class (Cell 26).

Observations: Some classes have lower recall due to limited features. Increasing features improves performance but with diminishing returns.
   
### 6. CIFAR-10 Ensemble Classifier

https://github.com/aspabhi31/PortfolioProjects/blob/main/19.%20Project-Cifar10/Solution%20(1).ipynb

This project uses PCA and an ensemble of classifiers (Random Forest, Logistic Regression, KNN, SVM) on the CIFAR-10 dataset.

#### Requirements
- Python 3
- Libraries: cifar10, numpy, matplotlib, scikit-learn

#### Usage
Run the notebook to train models and generate predictions.

#### Results
Ensemble accuracy: ~99% (as per notebook output).

### 7. Twitter Sentiment Analysis on US Airline Tweets

https://github.com/aspabhi31/PortfolioProjects/blob/main/Copy_of_TwitterSentimentAnalysisonUSAirlineTweets_Commented%20(1).ipynb

This project performs sentiment analysis on tweets about US airlines to classify them as positive, negative, or neutral using Natural Language Processing (NLP) techniques and machine learning. The dataset contains tweets from passengers, and the goal is to predict sentiment based on tweet content.

#### Project Overview

- **Objective**: Classify the sentiment of tweets about US airlines (positive, negative, neutral) using a Multinomial Naive Bayes classifier.
- **Dataset**: 
  - `train.csv`: Contains tweets with labeled sentiments (positive, negative, neutral).
  - `test.csv`: Contains tweets for which sentiments are predicted.
- **Technologies**: Python, Pandas, NLTK, Scikit-learn.
- **Methodology**: 
  - Preprocess tweets by tokenizing, removing stopwords, and lemmatizing.
  - Convert text to numerical features using CountVectorizer with unigrams, bigrams, and trigrams.
  - Train a Multinomial Naive Bayes model and predict sentiments for the test set.
- **Output**: Predictions saved to `pred.csv`

### 8. Distracted Driver Detection

https://github.com/aspabhi31/PortfolioProjects/blob/main/Distracted%20driver%20detection.ipynb

This project uses a CNN built with Keras/TensorFlow to classify driver images into 10 distraction categories (c0-c9) from the State Farm Kaggle competition.

#### Dataset
- From: https://www.kaggle.com/c/state-farm-distracted-driver-detection
- Classes: c0 (safe driving), c1 (texting right), etc.

#### Model
- Simple CNN with Conv2D, MaxPooling, and Dense layers.
- Trained using ImageDataGenerator on the train set.
- Predictions on test images saved to `predictions.txt`.

#### How to Run
1. Download the dataset.
2. Update paths in the notebook.
3. Run in Jupyter or Colab.

#### Requirements
- TensorFlow/Keras
- Pandas, NumPy, etc.

### 9. Neural Machine Translation (French to English)

https://github.com/aspabhi31/PortfolioProjects/blob/main/Neural_Machine_Translation(French_to_English).ipynb

This project implements a neural machine translation model using TensorFlow to translate French sentences to English. The model uses an LSTM-based sequence-to-sequence architecture.

#### Requirements
- Python 3
- TensorFlow
- NumPy
- NLTK
- scikit-learn

#### Dataset
The dataset (`fra.txt`) is sourced from http://www.manythings.org/anki/.

#### Usage
1. Install dependencies: `pip install tensorflow numpy nltk scikit-learn`
2. Place `fra.txt` in the project directory.
3. Run the Jupyter Notebook: `jupyter notebook Neural_Machine_Translation(French_to_English).ipynb`

#### Model Details
- Encoder-Decoder LSTM architecture
- Trained on 100 samples for 30 epochs
- Designed an encoder-decoder architecture with embedding and LSTM layers, achieving a validation accuracy of 87.32%
  
### 10. Case Study of U Food Marketing
https://github.com/aspabhi31/PortfolioProjects/blob/main/Python%20Project-%20U%20Food%20Marketing%20Analysis.ipynb


**Tools:** Python (Pandas, Seaborn, Matplotlib)  
- Cleaned and preprocessed a “dirty” marketing dataset using Pandas.
- Consolidated redundant columns to optimize the dataframe.
- Explored correlations between various features and accepted campaigns.
- Visualized insights with heatmaps, regplots, barplots, and pointplots.
- Utilized group bys, value counts, and sorting for meaningful summaries.

---

### 11. Exploratory Data Analysis on US Household Income
https://github.com/aspabhi31/PortfolioProjects/blob/main/SQL-%20US%20Household%20income%20Data%20Cleaning.sql
https://github.com/aspabhi31/PortfolioProjects/blob/main/SQL-%20US%20Household%20Income%20Data%20Exploration.sql


**Tools:** SQL, MySQL  
- Performed structured data cleaning, including duplicate removal with window functions.
- Standardized inconsistent or missing values using grouping and transformations.
- Conducted statistical analyses such as Median and Mean Salary using JOINs.
- Provided actionable socioeconomic insights from household income data.

---

### 12. ETL Pipelines and Data Transformation on Azure
**Tools:** Azure Data Factory, Storage Accounts, Azure Data Studio, Azure SQL Database  
- Designed and executed end-to-end ETL pipelines using Azure tools.
- Transformed and cleaned data (JOINs, Filters) within Azure SQL Database.
- Automated data movement between Azure components, exporting to CSV.
- Built and triggered pipelines integrating storage accounts and SQL services.

---

### 13. US Debt Tracker
https://github.com/aspabhi31/PortfolioProjects/blob/main/US%20Debt%20Tracker%20Project.xlsx


**Tools:** Microsoft Excel  
- Preprocessed debt data by transposing, filling nulls, and applying tables.
- Utilized forecasting functions, pivot tables, and pivot charts for trend analysis.
- Answered business-critical questions on year-over-year percentage change, largest monthly swings, and future debt forecasts.

---

### 14. Advanced Data Visualizations (Rental Properties Analysis)
https://github.com/aspabhi31/PortfolioProjects/blob/main/Data%20Visualization-%20Rental%20Properties%20Analysis%20Dashboard%20for%20StayCatin.com.txt


**Tools:** Tableau  
- Loaded rental property data from Excel into Tableau.
- Used calculated fields, bar graphs, trends, scatter plots, tables, and maps.
- Built interactive dashboards, uncovering insights on neighborhoods, property types, and market dynamics.

---

### 15. Case Study of Startup Funding in India
https://github.com/aspabhi31/PortfolioProjects/blob/main/Startups%20funding%20Case%20Study%20Python%20Project%20Part%201.ipynb
https://github.com/aspabhi31/PortfolioProjects/blob/main/Startups%20funding%20Case%20Study%20Python%20Project%20Part%202.ipynb


**Tools:** Python (Pandas, NumPy, Matplotlib)  
- Cleaned and preprocessed a complex startup funding dataset.
- Identified key regions, investor patterns, and top industries for Indian startups.
- Created visualizations (pie charts, bar graphs, trend lines) and performed comparative group analysis by investment type.
### 16. BERT-based SMS Spam Classifier

A **BERT-powered binary text classifier** to detect spam in SMS messages using Hugging Face Transformers. This project fine-tunes the `bert-base-uncased` model on the **SMS Spam Collection Dataset** from UCI.

https://github.com/aspabhi31/PortfolioProjects/blob/main/bert_spam_classification%20(1).ipynb

---
#### Dataset

- **Source**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Original Size**: 5,572 messages
- **Labels**: `ham` (0), `spam` (1)
- **Class Distribution (Original)**:
  - Ham: 4,825 (86.6%)
  - Spam: 747 (13.4%)
- **Balanced Subset Used**: 1,000 `ham` + 747 `spam` = **1,747 samples**

> Dataset file: `spam.csv` (included)

---

#### Project Overview

This notebook demonstrates:
- Loading and preprocessing SMS data
- Handling class imbalance via undersampling
- Tokenization with `BertTokenizer`
- Fine-tuning `BertModel` for binary classification
- Training with PyTorch and Hugging Face
- Evaluation using accuracy, precision, recall, F1

---
### 17. Vehicle Damage Detection Project

https://github.com/aspabhi31/PortfolioProjects/tree/main/Project-Car%20Damage%20Detection

A deep learning-based multi-class image classifier for detecting vehicle damage using PyTorch and a fine-tuned ResNet50 model. The project identifies damage types and locations (front/rear) in vehicle images, categorizing them as normal, breakage, or crushed.

This repository includes two Jupyter notebooks:
- **damage_prediction.ipynb**: Main training, evaluation, and prediction pipeline.
- **hyperparameter_tunning.ipynb**: Hyperparameter optimization using Optuna to find the best learning rate and dropout values.

---

#### Dataset

- **Source**: Custom dataset (not publicly available; assumed to be in `./dataset` folder). If you have your own dataset, structure it similarly.
- **Classes** (6 total):
  - F_Breakage (Front Breakage)
  - F_Crushed (Front Crushed)
  - F_Normal (Front Normal)
  - R_Breakage (Rear Breakage)
  - R_Crushed (Rear Crushed)
  - R_Normal (Rear Normal)
- **Size**: 2,300 images.
- **Split**: 75% training (1,725 images), 25% validation (575 images).
- **Preprocessing**: Augmentations include random horizontal flip, rotation (10°), color jitter, resize to 224x224, normalization (ImageNet stats).

> **Note**: Place your dataset in a `./dataset` folder with subfolders for each class. If the dataset is large, consider using Git LFS or linking to an external source (e.g., Google Drive or Kaggle).

---

#### Project Overview

- **Model**: Pre-trained ResNet50 (from `torchvision.models`), with frozen layers except `layer4` and a custom fully connected head (dropout + linear layer).
- **Fine-Tuning**: Only trainable layers are optimized using Adam.
- **Hyperparameter Tuning** (in `hyperparameter_tunning.ipynb`): Uses Optuna to optimize learning rate (lr) and dropout rate. Best params from example runs: lr ≈ 0.0004–0.005, dropout ≈ 0.2–0.66 (results may vary due to randomness).
- **Training** (in `damage_prediction.ipynb`): 10 epochs, CrossEntropyLoss, Adam optimizer.
- **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix.
- **Prediction**: Example inference on new images.

---

## Technical Skills Used

- Languages: Python, Java, SQL (Postgres, MySQL), NoSQL (MongoDB)
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow, Keras
- Techniques: Data preprocessing and EDA, feature engineering, model training, statistical analysis, data visualization

---

## About Me

**Abhijeet Singh Pawar**  
- Master of Applied Science in Computer Engineering, Memorial University of Newfoundland (2022–2024)
- Data Analyst, Coding Expert, and AI Enthusiast with proven internship and tutoring experience
- Contact: aspabhi31@gmail.com  
- [LinkedIn](https://linkedin.com/in/abhijeet-singh-pawar-482576149/) | [GitHub](https://github.com/aspabhi31)

---

## How to Use

- Each project folder contains source code, instructions, and (where possible) datasets or links to them.
- For setup or usage instructions, see the README provided in each project directory.

---

## License

Projects are provided for educational and demonstrative purposes. Contact the author for use or collaboration.

---
