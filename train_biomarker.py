import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

print("Training biomarker model...")

data_path = os.path.join("Biomarker", "Supplementary data 3.xlsx")
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}")
    exit(1)

# Read data
data = pd.read_excel(data_path)

# Clean data (remove > and <)
for col in data.columns:
    data[col] = data[col].astype(str).str.replace('>', '', regex=False)
    data[col] = data[col].astype(str).str.replace('<', '', regex=False)

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Selected top 10 features based on previous training
top_10_features = ['HE4', 'CA125', 'LYM%', 'Age', 'LYM#', 'AST', 'NEU', 'CA19-9', 'CL', 'ALP']

# Extract features and target
X = data[top_10_features]
y = data["TYPE"]

# Drop missing values if any
valid_indices = X.dropna().index
X = X.loc[valid_indices]
y = y.loc[valid_indices]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Check Accuracy
pred = model.predict(X_test)
print("Accuracy of top 10 features model:", accuracy_score(y_test, pred))

# Save Model
joblib.dump(model, "ovarian_model_top10.pkl")
print("Model saved as ovarian_model_top10.pkl")
