import joblib

model = joblib.load("rf_ids_model.pkl")

print(model.feature_names_in_)