# Height vs. Weight Prediction using Linear Regression

This project demonstrates a linear regression model to predict weight based on height using a dataset of 10,000 samples. The implementation includes data visualization, preprocessing, training, evaluation, and prediction.

## Features
- Data visualization using scatter plots and pair plots.
- Data preprocessing (dropping non-numerical columns and standardization).
- Linear regression implementation.
- Model evaluation using performance metrics (MAE, MSE, R², Adjusted R²).
- Prediction for new data points.
- OLS regression analysis.

---

## Setup Instructions

### Prerequisites
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `statsmodels`

### Installation
1. Clone this repository or copy the code into a Jupyter notebook or any Python environment.
2. Install the required libraries using:

   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

## Usage
**1. Load the Dataset**
The dataset weight-height 2.csv should be available in the path /content/weight-height 2.csv. Update the path if necessary.
  df = pd.read_csv('/path/to/weight-height 2.csv')
  
**2. Run the Script**
Execute the cells sequentially in a Jupyter notebook or any Python IDE.

**3. Output**
  . Visualizations of the data and best-fit line.
  
  . Performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), R², and Adjusted R².
  
  . Predictions for the test dataset and new height values.

## Workflow
**1. Data Exploration**
  . Load and inspect the dataset.
  . Visualize height vs. weight using a scatter plot.
**2. Preprocessing**
  . Drop non-numerical columns (Gender).
  . Compute correlations and visualize data relationships using a pair plot.
**3. Train-Test Split**
  . Split the dataset into training (75%) and testing (25%) subsets.
**4. Standardization**
  . Standardize the Height values to improve model performance.
**5. Model Training**
  . Train a linear regression model using the Height feature to predict Weight.
**6. Model Evaluation**

    **Evaluate the model using**:
      . Mean Absolute Error (MAE)
      . Mean Squared Error (MSE)
      . R² (Coefficient of Determination)
      . Adjusted R²
    **Perform OLS regression analysis for detailed statistics.**
    
**7. Prediction**
  . Make predictions on the test data.
  . Predict weight for a new height value (e.g., 69 inches).
  
## Example Results
# Model Performance
  . Mean Absolute Error: 9.72
  . Mean Squared Error: 149.31
  . R-squared: 0.857
  . Adjusted R-squared: 0.857
# Prediction
  . Weight for height 69 inches: 181.65 lbs
  
## Acknowledgments
Dataset Source: [Synthetic height-weight dataset]
Model Implementation: Scikit-learn, Statsmodels
Visualizations: Matplotlib, Seaborn

## References
  Scikit-learn Documentation
  Statsmodels Documentation

  
This `README.md` provides an organized overview of your project, including setup instructions, workflow, and example results. Update the dataset source link in the acknowledgments section if required.
