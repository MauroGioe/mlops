import mlflow
import pandas as pd
import pickle
import nannyml as nml
from sklearn.metrics import f1_score

model_name = "xgboost"
model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

df = pd.read_csv("./data/prod.csv")
df = df.sample(frac=0.50)

y = df["Class"]
X = df.drop("Class", axis=1)
y_predicted_prod = model.predict(X)
y_predicted_proba_prod = model.predict_proba(X)[:,1]
prod_with_preds = X.assign(y_pred=y_predicted_prod, y_pred_proba=y_predicted_proba_prod, Class=y)

f1_score(prod_with_preds["Class"],prod_with_preds["y_pred"])

test = pd.read_csv("./data/test.csv")
y_test = test["Class"]
X_test = test.drop("Class", axis=1)
y_predicted_test = model.predict(X_test)
y_predicted_proba_test = model.predict_proba(X_test)[:,1]
test_with_preds = X_test.assign(y_pred=y_predicted_test, y_pred_proba=y_predicted_proba_test, Class=y_test)

with open("top_features.pkl", "rb") as f:
    top_features =  pickle.load(f)

udc = nml.UnivariateDriftCalculator(column_names=top_features, chunk_number=6)


udc.fit(test)

univariate_data_drift = udc.calculate(X)
figure = univariate_data_drift.plot(kind='drift')
#figure.write_image("./univariate drift.png", format="png")
#kaleido bug doesn't work with windows in V0.2
figure.show()

estimator = nml.CBPE(y_pred_proba='y_pred_proba', y_pred='y_pred', y_true='Class', metrics=['f1'],
                     problem_type='classification_binary', chunk_number=6)

estimator.fit(test_with_preds)
results = estimator.estimate(prod_with_preds)
metric_fig = results.plot()
#metric_fig.write_image("./perfomance estimation.png")
#kaleido bug doesn't work with windows in V0.2
metric_fig.show()