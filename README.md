# Student Dropout Prediction Dataset & Model

## 📊 Dataset Description
This dataset simulates undergraduate students and their likelihood of dropping out in the first year.

### Variables
- **age (int)**: Student age (16–30 typical, outliers at 10, 80).
- **gender (categorical)**: Male, Female, Other.
- **place_of_origin (categorical)**: Urban, Rural, International.
- **hs_avg (float)**: High school average (40–100, with some outliers).
- **admission_test (float)**: Entrance exam score (200–800, outliers at 50, 1200).
- **first_semester_gpa (float)**: GPA (0–5, with outliers -1, 6).
- **socioeconomic_level (categorical)**: Low, Medium, High.
- **scholarship (binary)**: 1 = yes, 0 = no.
- **loan (binary)**: 1 = yes, 0 = no.
- **dropout (binary categorical)**: Yes / No.

### Null Values
- Introduced randomly in ~5% of rows for each column.

### Outliers
- **Age**: unrealistic values (10, 80).  
- **GPA**: invalid values (-1, 6).  
- **Admission test**: unrealistic scores (50, 1200).  

## ⚙️ Model
We apply **Logistic Regression** as a supervised learning method to predict dropout likelihood.  
Steps:
1. Handle missing values (numeric → mean, categorical → mode).
2. Encode categorical variables.
3. Train/test split (80/20).
4. Logistic regression training and evaluation.

## 📂 Project Structure
```
student_dropout_prediction/
│── dataset/
│   └── students_dropout.csv
│── src/
│   └── generate_dataset.py
│   └── model_training.py
│── README.md
```
