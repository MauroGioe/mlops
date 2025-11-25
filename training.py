import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.inspection import permutation_importance
import mlflow
import pickle
import subprocess


experiment_name = "Credit card fraud"
current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
experiment_id=current_experiment['experiment_id']
subprocess.run(["mlflow","experiments","delete","--experiment-id",f"{experiment_id}"])


mlflow.set_experiment(experiment_name)
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

y_train = train["Class"]
X_train = train.drop("Class", axis=1)

y_test = test["Class"]
X_test = test.drop("Class", axis=1)

scale_pos_weight = y_train.value_counts()[0]/y_train.value_counts()[1]

param_grid = {
    'max_depth': [x for x in range(2,20)],
    'learning_rate': [0.01, 0.5, 1],
    'n_estimators': [10, 100, 200]
}
model = XGBClassifier(objective='binary:logistic', seed = 123, scale_pos_weight=scale_pos_weight)
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, scoring ="f1", cv=3)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
with open("./models/xgb.pkl", "wb") as f:
    pickle.dump(best_model, f)

results = pd.DataFrame(grid_search.cv_results_)
results = results[['params', 'mean_test_score', 'std_test_score']]
max_run = results["mean_test_score"].idxmax()
results = results.values
for i,run in enumerate(results):
    with mlflow.start_run(run_name="cv_"+str(i)):
        mlflow.set_tag("model", "xgboost")
        # Log model configuration/params
        mlflow.log_params(run[0])
        # Log metrics
        metrics = {
            'f1-score': run[1],
            'std': run[2]
        }
        mlflow.log_metrics(metrics)
        if i == max_run:
            mlflow.log_artifact("./models/xgb.pkl", artifact_path="models_pickle")
            y_predict = best_model.predict(X_test)
            f1 = f1_score(y_test, y_predict)
            precision = precision_score(y_test, y_predict)
            recall = recall_score(y_test, y_predict)
            best_metrics = {"test f1-score":f1, "test precision":precision, "test recall":recall}
            mlflow.log_metrics(best_metrics)
            mlflow.sklearn.log_model(sk_model=best_model, name="xgboost", input_example=X_train,
                                     registered_model_name="xgboost")

    mlflow.end_run()

perm_importance = permutation_importance(best_model, X_test, y_test, scoring = "f1")
sorted_idx = perm_importance.importances_mean.argsort()
list(zip(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx]))[::-1]
top_features = X_test.columns[sorted_idx][::-1][:4].to_list()
with open('top_features.pkl', 'wb') as f:
    pickle.dump(top_features, f)