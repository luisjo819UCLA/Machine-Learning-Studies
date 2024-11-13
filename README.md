# Machine Learning Class Repository

This repository contains projects and exercises completed as part of a Machine Learning course. The repository demonstrates the application of various machine learning techniques, data preprocessing steps, and model evaluation methods across different datasets. Each file represents a specific task or set of tasks, showcasing a variety of algorithms, machine learning tools, and data processing techniques.

## Contents

1. **Quarterly Earnings Analysis with Google Trends**
   - **Techniques**: Feature engineering, normalization, Lasso regression, hyperparameter tuning, cross-validation
   - **Machine Learning Tools**: `pytrends`, `scikit-learn`
   - **Description**: This project leverages Google Trends data to create features that could nowcast quarterly earnings for Apple. Using Lasso regression, it selects features based on importance, examining the impact of different regularization values (`λ`) on model performance.

2. **Taylor Rule Modeling and Polynomial Regression**
   - **Techniques**: Train-test split, Ordinary Least Squares (OLS) regression, polynomial regression, mean squared error (MSE) evaluation
   - **Machine Learning Tools**: `statsmodels`, `scikit-learn`
   - **Description**: This file builds a predictive model using the Taylor Rule, evaluating model performance for polynomial degrees of 1, 2, and 3. It demonstrates the trade-off between bias and variance through model complexity and calculates in-sample and out-sample MSEs to measure fit.

3. **Predictive Analysis of Apple Stock Premiums**
   - **Techniques**: Logistic regression, feature engineering, profit calculation, cumulative profit visualization
   - **Machine Learning Tools**: `yfinance`, `scikit-learn`, `matplotlib`
   - **Description**: This project creates a logistic regression model to predict Apple stock price movement. Features include the stock price difference, option premium, and monthly target outcome. Profits are calculated based on correct predictions, and the cumulative profit is visualized over time.

4. **Predicting Bank Marketing Campaign Success**
   - **Techniques**: Data encoding, oversampling (SMOTE), decision trees, bagging, boosting, ensemble learning, super learner model, confusion matrix, F1 score
   - **Machine Learning Tools**: `scikit-learn`, `imblearn`, `seaborn`, `matplotlib`
   - **Description**: This project uses decision trees, bagging, and boosting to predict the success of a bank marketing campaign. Additionally, a super learner model combines the base models to improve prediction accuracy. Model performance is evaluated using accuracy, confusion matrices, and F1 scores, providing insight into each model's strengths.

5. **Country Development Classification using KMeans Clustering**
   - **Techniques**: KMeans clustering, data normalization, elbow method, silhouette analysis, descriptive statistics
   - **Machine Learning Tools**: `scikit-learn`, `matplotlib`
   - **Description**: This project applies KMeans clustering to classify countries based on socioeconomic indicators. The optimal number of clusters is determined using the elbow method and silhouette scores, and countries are divided into clusters representing varying levels of development.

---

## Machine Learning Techniques and Tools Covered

- **Data Preprocessing**: Feature scaling, dummy variable creation, handling categorical data
- **Regression Models**: Ordinary Least Squares (OLS), Logistic Regression, Polynomial Regression, Lasso Regression
- **Classification Models**: Decision Trees, Bagging, Boosting, Super Learner models
- **Clustering Models**: KMeans Clustering, elbow method, silhouette analysis
- **Feature Engineering**: Creating meaningful variables from raw data, calculating returns, option premiums
- **Ensemble Learning**: Bagging, Boosting, Super Learner model
- **Evaluation Metrics**: Mean Squared Error (MSE), Accuracy, F1 Score, Confusion Matrix, Profit Calculation
- **Data Visualization**: Scatter plots, cumulative profit graphs, confusion matrices, clustering visualizations

## Tools Learned in this Course

- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning Libraries**: `scikit-learn`, `statsmodels`, `imblearn` for oversampling with SMOTE
- **Data Sources**: `yfinance` for stock data, Google Trends API (`pytrends`)
- **Data Preprocessing**: `StandardScaler`, feature encoding, handling class imbalance with SMOTE
- **Model Selection and Tuning**: Cross-validation, hyperparameter tuning for regularization (`λ`), selection of optimal number of clusters (elbow method)

---

## How to Use the Code

Each file is a self-contained Jupyter notebook or script. Clone the repository and install the necessary Python libraries using:

```bash
pip install -r requirements.txt
```

Then, open each file in Jupyter Notebook or run as a Python script. Ensure that all required datasets are available in the repository or in the correct file paths.

## Summary

This repository provides a comprehensive view of fundamental machine learning techniques and tools, illustrating the versatility of data science applications across finance, marketing, and economic classification. By working with a variety of data types and exploring multiple model approaches, this course has built a strong foundation in machine learning practices.
