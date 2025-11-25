import mlflow
import subprocess

experiment_name = "Credit card fraud"
try:
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']
    #subprocess.run(["mlflow","experiments","delete","--experiment-id",f"{experiment_id}"])
    mlflow.set_tracking_uri("192.168.0.1")
    subprocess.run(["mlflow","gc"])
except:
    print("Removal not executed")
