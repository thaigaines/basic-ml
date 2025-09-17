import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


## Load data
data = pd.read_csv('./data/cleaned_crime_data.csv')

# Split features (X) and target (y)
X = data[["Vict Age", "Vict Sex Num"]] # Double brackets = DataFrame (2D)
y = data["Crm Cd"]                     # Single bracket = Series (1D)

# Removing the mean and dividing by the standard deviation (standardization), or by rescaling to a fixed range (normalization).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets (default test_size=0.25, random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)


# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Coefficients: positive = positive correlation, negative = negative correlation
print(f"Feature 1 (Vict Age) coefficient: {model.coef_[0, 0]} Feature 2 (Vict Sex) coefficient: {model.coef_[0, 1]}")

# In this data set, there is no meaningful correlation between a victim's age and sex with the crime committed.