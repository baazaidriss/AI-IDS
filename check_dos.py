import joblib
import pandas as pd

# Load model and encoder
model = joblib.load('rf_ids_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# Load DoS test data
df = pd.read_csv('real_dos_test.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_per_').str.replace(r'[^a-z0-9_]', '', regex=True)

# Get expected columns
expected = list(model.feature_names_in_)
df = df[expected]

# Predict
pred = model.predict(df)
proba = model.predict_proba(df)

print('Predictions:', encoder.inverse_transform(pred))
print()

for i, p in enumerate(proba):
    print(f'Row {i}:')
    for j, cls in enumerate(encoder.classes_):
        print(f'  {cls}: {p[j]*100:.1f}%')
    print()