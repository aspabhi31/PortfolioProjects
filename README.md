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

### 5. Case Study of U Food Marketing
https://github.com/aspabhi31/PortfolioProjects/blob/main/Python%20Project-%20U%20Food%20Marketing%20Analysis.ipynb


**Tools:** Python (Pandas, Seaborn, Matplotlib)  
- Cleaned and preprocessed a “dirty” marketing dataset using Pandas.
- Consolidated redundant columns to optimize the dataframe.
- Explored correlations between various features and accepted campaigns.
- Visualized insights with heatmaps, regplots, barplots, and pointplots.
- Utilized group bys, value counts, and sorting for meaningful summaries.

---

### 6. Exploratory Data Analysis on US Household Income
https://github.com/aspabhi31/PortfolioProjects/blob/main/SQL-%20US%20Household%20income%20Data%20Cleaning.sql
https://github.com/aspabhi31/PortfolioProjects/blob/main/SQL-%20US%20Household%20Income%20Data%20Exploration.sql


**Tools:** SQL, MySQL  
- Performed structured data cleaning, including duplicate removal with window functions.
- Standardized inconsistent or missing values using grouping and transformations.
- Conducted statistical analyses such as Median and Mean Salary using JOINs.
- Provided actionable socioeconomic insights from household income data.

---

### 7. ETL Pipelines and Data Transformation on Azure
**Tools:** Azure Data Factory, Storage Accounts, Azure Data Studio, Azure SQL Database  
- Designed and executed end-to-end ETL pipelines using Azure tools.
- Transformed and cleaned data (JOINs, Filters) within Azure SQL Database.
- Automated data movement between Azure components, exporting to CSV.
- Built and triggered pipelines integrating storage accounts and SQL services.

---

### 8. US Debt Tracker
https://github.com/aspabhi31/PortfolioProjects/blob/main/US%20Debt%20Tracker%20Project.xlsx


**Tools:** Microsoft Excel  
- Preprocessed debt data by transposing, filling nulls, and applying tables.
- Utilized forecasting functions, pivot tables, and pivot charts for trend analysis.
- Answered business-critical questions on year-over-year percentage change, largest monthly swings, and future debt forecasts.

---

### 9. Advanced Data Visualizations (Rental Properties Analysis)
https://github.com/aspabhi31/PortfolioProjects/blob/main/Data%20Visualization-%20Rental%20Properties%20Analysis%20Dashboard%20for%20StayCatin.com.txt


**Tools:** Tableau  
- Loaded rental property data from Excel into Tableau.
- Used calculated fields, bar graphs, trends, scatter plots, tables, and maps.
- Built interactive dashboards, uncovering insights on neighborhoods, property types, and market dynamics.

---

### 10. Case Study of Startup Funding in India
https://github.com/aspabhi31/PortfolioProjects/blob/main/Startups%20funding%20Case%20Study%20Python%20Project%20Part%201.ipynb
https://github.com/aspabhi31/PortfolioProjects/blob/main/Startups%20funding%20Case%20Study%20Python%20Project%20Part%202.ipynb


**Tools:** Python (Pandas, NumPy, Matplotlib)  
- Cleaned and preprocessed a complex startup funding dataset.
- Identified key regions, investor patterns, and top industries for Indian startups.
- Created visualizations (pie charts, bar graphs, trend lines) and performed comparative group analysis by investment type.
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
