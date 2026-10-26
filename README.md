# Multi-Disease Prediction System

A machine learning project built with Python and Streamlit to predict outcomes for five datasets.

## Prediction Modules

- Diabetes Prediction
- Heart Disease Prediction
- Chronic Kidney Disease Prediction
- Breast Cancer Prediction
- Thyroid Cancer Recurrence Prediction

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit

## Run the Project

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Project Structure

```text
Multi_Disease_Prediction_System/
├── app.py
├── README.md
├── requirements.txt
├── datasets/
├── models/
└── utils/
```

## Note

This project is developed for academic purposes. Predictions are based on the trained models and datasets and should not be used as a substitute for professional medical advice.

The thyroid module predicts recurrence in the thyroid cancer dataset; it does not detect thyroid disease in general.
