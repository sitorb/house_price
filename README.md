# House Price Prediction using Decision Tree Regression

## Overview

This project aims to predict house prices using a machine learning model called Decision Tree Regression. The model is trained on a dataset containing various features of houses, such as the number of bedrooms, bathrooms, square footage, and location.

## Code Structure and Logic

1. **Data Loading and Preprocessing:**
   - The code begins by importing necessary libraries like pandas, numpy, scikit-learn, and matplotlib.
   - Training and testing data are loaded from CSV files (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`).
   - Data is checked for missing values and duplicates, which are handled accordingly.
   - Categorical variables are converted to numerical representations using Label Encoding for consistency.
   - Target variables (house prices) are converted to numeric format.
   - Data types are validated to ensure all features are numerical.

2. **Hyperparameter Tuning:**
   - `GridSearchCV` is employed to find the best hyperparameters for the Decision Tree model, optimizing for negative mean squared error.
   - This helps in selecting the most suitable values for parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`.

3. **Model Training and Evaluation:**
   - The best model obtained from hyperparameter tuning is trained on the training data.
   - Predictions are made on the test data, and performance metrics like Mean Squared Error (MSE) and R-squared are calculated.

4. **Visualizations:**
   - Various visualizations are created to gain insights into the model and data:
     - **Feature Importance Plot:** Shows the relative importance of different features in predicting house prices.
     - **Actual vs. Predicted Values Plot:** Compares the actual and predicted house prices.
     - **Residual Plot:** Helps assess the model's error distribution.
     - **Learning Curve:** Illustrates the model's performance as the training data size increases.
     - **Prediction Distribution Plot:** Shows the distribution of predicted house prices.
     - **Residual Histogram:** Displays the distribution of residuals.
     - **Heatmap:** Shows the correlation between different features.
     - **Decision Tree Visualization:** Provides a visual representation of the decision tree structure.
![image](https://github.com/user-attachments/assets/6bd12dea-7f2d-4a87-8536-ea2c2c810b59)

![image](https://github.com/user-attachments/assets/3a59c8f7-84d6-4d6d-8ab0-7a5f54e08b3d)
![image](https://github.com/user-attachments/assets/e210a9b5-64da-4965-99f7-42da39e071a9)


## Technology and Algorithms

**Technology:**
- **Python:** The primary programming language used for this project.
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn.
- **Google Colab:** The development environment where the code is executed.

**Algorithms:**
- **Decision Tree Regression:** A supervised learning algorithm used to predict continuous target variables (house prices in this case). It creates a tree-like structure to make decisions based on features.
- **Label Encoding:** A technique used to convert categorical variables into numerical form.
- **Grid Search:** An optimization technique used to find the best hyperparameters for a model.


## Conclusion

This project demonstrates the application of Decision Tree Regression for predicting house prices. The model's performance is evaluated using relevant metrics, and various visualizations provide insights into its behavior.
