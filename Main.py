import pickle

# Load the Gradient Boosting model
with open("best_gb_model.pkl", "rb") as file:
    loaded_gb = pickle.load(file)

# Load the Random Forest model
with open("best_rf_model.pkl", "rb") as file:
    loaded_rf = pickle.load(file)

# Load the Logistic Regression model
with open("best_lr_model.pkl", "rb") as file:
    loaded_lr = pickle.load(file)
