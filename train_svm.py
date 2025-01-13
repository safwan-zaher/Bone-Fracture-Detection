import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('C:\\Users\\WALTON\\Desktop\\FinalProject\\extracted_features_parallel.csv')


X = df[['Correlation', 'Energy', 'Homogeneity', 'Contrast', 'Dissimilarity']]
y = df['label']
scaler = StandardScaler()
X = scaler.fit_transform(X)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, refit=True, verbose=1, cv=5)
grid_rf.fit(X_train, y_train)

# Save the best Random Forest model
joblib.dump(grid_rf.best_estimator_, 'rf_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')


y_pred_rf = grid_rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
