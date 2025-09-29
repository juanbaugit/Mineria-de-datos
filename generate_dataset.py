import pandas as pd
import numpy as np
import random

np.random.seed(42)
n = 500

age = np.random.randint(16, 30, size=n).astype(float)
gender = np.random.choice(['Male', 'Female', 'Other'], size=n)
origin = np.random.choice(['Urban', 'Rural', 'International'], size=n)
hs_avg = np.random.normal(75, 10, size=n)
admission_test = np.random.normal(600, 100, size=n)
first_semester_gpa = np.random.normal(3, 0.7, size=n)
socioeconomic_level = np.random.choice(['Low', 'Medium', 'High'], size=n)
scholarship = np.random.choice([0, 1], size=n)
loan = np.random.choice([0, 1], size=n)

dropout = []
for i in range(n):
    prob = 0.3
    if hs_avg[i] < 65 or first_semester_gpa[i] < 2.5:
        prob += 0.3
    if socioeconomic_level[i] == 'Low' and scholarship[i] == 0:
        prob += 0.2
    dropout.append(np.random.choice(['Yes', 'No'], p=[prob, 1 - prob]))

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'place_of_origin': origin,
    'hs_avg': hs_avg,
    'admission_test': admission_test,
    'first_semester_gpa': first_semester_gpa,
    'socioeconomic_level': socioeconomic_level,
    'scholarship': scholarship,
    'loan': loan,
    'dropout': dropout
})

for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

df.loc[random.sample(range(n), 5), 'age'] = np.random.choice([10, 80], size=5)
df.loc[random.sample(range(n), 5), 'first_semester_gpa'] = np.random.choice([6, -1], size=5)
df.loc[random.sample(range(n), 5), 'admission_test'] = np.random.choice([50, 1200], size=5)

df.to_csv("dataset/students_dropout.csv", index=False)
print("✅ Dataset generated!")
