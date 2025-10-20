# 23AD054_EDA_ASSIGNMENT_2

----Student Performance Prediction Using Deep Learning----

Project Overview:
This project focuses on predicting the final grades of Portuguese students using machine learning and deep learning techniques. It leverages the Portuguese Student Performance Dataset (student-por.csv) from the UCI Machine Learning Repository. The study combines Exploratory Data Analysis (EDA), feature engineering, and a Sequential Multi-Layer Perceptron (MLP) model to identify key factors influencing student performance and predict academic outcomes.

Key Features:
Analysis of demographic, behavioral, family, and academic features.
Preprocessing includes handling missing values, outliers, categorical encoding, and feature scaling.
Model input features are optimized using Principal Component Analysis (PCA) for dimensionality reduction.
Predictive modeling with a Deep Learning MLP, capable of classifying performance into Low, Medium, and High categories or predicting final grades numerically.
Evaluation using metrics such as RMSE, MAE, R², and Accuracy for regression and classification.
Data visualization includes histograms, scatter plots, boxplots, correlation heatmaps, and more to provide insights into student performance patterns.

Dependencies:
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow / keras
joblib

Install dependencies using:
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib

Usage:
Extract the dataset ZIP file (e.g., student-por.zip).
Run the Python scripts to perform:
Data loading & preprocessing
Exploratory Data Analysis (EDA)
Model training and evaluation
Visualization of results

The trained MLP model and preprocessing pipeline are saved as:
student_mlp_regressor.h5 (model)
preprocessing_pipeline.joblib (pipeline)

Results:
The MLP model achieved strong predictive performance with Test RMSE ≈ 1.42, MAE ≈ 0.90, R² ≈ 0.79.
Key predictors of student performance include G1, G2, study time, and absences.
Visualizations highlight relationships between features and final grades.

Future Enhancements:
Incorporate psychological and behavioral features for richer prediction.
Compare with ensemble models like XGBoost or Random Forest.
Build interactive dashboards for real-time monitoring of student performance.
Extend analysis to larger multi-school datasets for better generalization.
Implement Explainable AI techniques (SHAP/LIME) to interpret model predictions.

References:
UCI Machine Learning Repository – Student Performance Dataset
Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
Chollet, F. (2018). Deep Learning with Python. Manning Publications.
Raschka, S., & Mirjalili, V. (2019). Python Machine Learning. Packt Publishing.
