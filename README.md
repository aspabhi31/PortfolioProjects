Welcome to my GitHub repository showcasing personal projects in computer engineering, data science, and machine learning. These projects demonstrate my skills in Python programming, data analysis, image processing, and predictive modeling. Each project includes a brief description, technologies used, and key implementations based on my work.
Feel free to explore the code, provide feedback, or reach out via LinkedIn or email.

Table of Contents

1. Automatic License Plate Detection (ALPD) https://github.com/aspabhi31/PortfolioProjects/blob/main/Automatic%20License%20Plate%20Detection(ALPD).ipynb)      https://github.com/aspabhi31/PortfolioProjects/blob/main/Automatic%20License%20Plate%20Detection(ALPD).pdf


2. Combined Cycle Power Plant Energy Output Prediction 
https://github.com/aspabhi31/PortfolioProjects/tree/main/5.%20Project-Gradient%20Descent

3. Titanic Survival Prediction using Logistic Regression 
https://github.com/aspabhi31/PortfolioProjects/tree/main/7.%20Project-Logistic%20Regression/Logistic%20Regression-Titanic%20Dataset 

4. Technical Skills

5. Contact


Automatic License Plate Detection (ALPD)

Description

This project involves designing and implementing an image processing pipeline for automatic license plate detection. It handles diverse lighting and angle conditions to accurately identify and isolate license plates from images.

Technologies

Python
NumPy
Matplotlib
OpenCV

Key Features

Applied Gaussian blurring for noise reduction.
Used histogram equalization for contrast enhancement.
Implemented adaptive thresholding to manage uneven illumination.
Developed contour detection and filtering using geometric criteria (area, aspect ratio, shape) to isolate candidate license plate regions.
Optimized parameters like Gaussian kernel size, threshold block size, and contour area thresholds for improved detection accuracy.

How to Run

Clone the repository.
Install dependencies: pip install numpy matplotlib opencv-python.
Run the main script: python alpd.py (assuming images are in the data/ folder).

Combined Cycle Power Plant Energy Output Prediction

Description

This project predicts the net hourly electrical energy output of a combined cycle power plant using ambient variables. It uses a custom Gradient Descent algorithm built from scratch on a dataset of 9,568 data points collected over 6 years.

Technologies

Python
NumPy
scikit-learn (for scaling)

Key Features

Implemented multi-feature linear regression, including data loading, cost function computation, and parameter optimization.
Experimented with learning rates (e.g., 0.0001) and iterations (up to 1,000) to minimize mean squared error.
Incorporated feature scaling with StandardScaler to prevent overflow and enhance convergence.
Generated predictions on test data and exported to CSV.

How to Run

Clone the repository.
Install dependencies: pip install numpy scikit-learn.
Place the dataset in the data/ folder.
Run the main script: python power_plant_prediction.py.

Titanic Survival Prediction using Logistic Regression

Description

A binary classification model to predict passenger survival on the Titanic. It uses a dataset of 668 training samples and 223 test samples, with feature engineering to improve accuracy.

Technologies

Python
Pandas
NumPy
scikit-learn

Key Features

Extensive data preprocessing: handled missing values (e.g., imputed median Age, filled Embarked with mode), encoded categorical variables (Sex as binary, Embarked as ordinal), converted Cabin to binary indicator, and dropped irrelevant columns.
Extracted titles from names for enhanced feature engineering.
Trained Logistic Regression model with 'saga' solver, high max iterations, and low tolerance.
Generated predictions for the test set and exported to CSV.
Explored ensemble methods like Random Forest Classifier.

How to Run

Clone the repository.
Install dependencies: pip install pandas numpy scikit-learn.
Place the Titanic dataset (train.csv, test.csv) in the data/ folder.
Run the main script: python titanic_prediction.py.

Technical Skills

Languages: Java, Python, SQL (Postgres, MySQL), NoSQL (MongoDB)
Techniques: EDA, Statistical Analysis, Data Visualization
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras
Certification: Data Science and Machine Learning, Coding Ninjas, India

Contact

Email: aspabhi31@gmail.com
LinkedIn: abhijeet-singh-pawar-482576149
GitHub: aspabhi31

If you have any questions or suggestions, feel free to open an issue or pull request!
