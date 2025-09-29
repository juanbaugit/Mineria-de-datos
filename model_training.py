import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("dataset/students_dropout.csv")

for col in df.select_dtypes(include=['float64', 'int64']):
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)

le = LabelEncoder()
for col in ['gender', 'place_of_origin', 'socioeconomic_level', 'dropout']:
    df[col] = le.fit_transform(df[col])

X = df.drop('dropout', axis=1)
y = df['dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("✅ Logistic Regression Results")
print(classification_report(y_test, y_pred))
