import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from time import gmtime, strftime
from sklearn.metrics import roc_curve, auc
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt
import tarfile
import pickle as pkl
import boto3
import os
import sagemaker
import json

def download_from_s3(s3_client, local_file_path, bucket_name, s3_file_path):
    try:
        # Download the file
        s3_client.download_file(bucket_name, s3_file_path, local_file_path)
        print(f"File downloaded successfully to {local_file_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            print(f"An error occurred: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def upload_to_s3(s3_client, local_file_path, bucket_name, s3_file_path=None):
    # If S3 file path is not specified, use the basename of the local file
    if s3_file_path is None:
        s3_file_path = os.path.basename(local_file_path)

    try:
        # Upload the file
        s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"File {local_file_path} uploaded successfully to {bucket_name}/{s3_file_path}")
        return True
    except ClientError as e:
        print(f"ClientError: {e}")
        return False
    except FileNotFoundError:
        print(f"The file {local_file_path} was not found")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
        
def write_params(s3_client, step_name, params, notebook_param_s3_bucket_prefix):
    local_file_path = f"{step_name}.json"
    with open(local_file_path, "w") as f:
        f.write(json.dumps(params))
    base_local_file_path = os.path.basename(local_file_path)
    bucket_name = notebook_param_s3_bucket_prefix.split("/")[2] # Format: s3://<bucket_name>/..
    s3_file_path = os.path.join("/".join(notebook_param_s3_bucket_prefix.split("/")[3:]),  base_local_file_path)
    upload_to_s3(s3_client, local_file_path, bucket_name, s3_file_path)
    
def read_params(s3_client, notebook_param_s3_bucket_prefix, step_name):
    local_file_path = f"{step_name}.json"
    base_local_file_path = os.path.basename(local_file_path)
    bucket_name = notebook_param_s3_bucket_prefix.split("/")[2] # Format: s3://<bucket_name>/..
    s3_file_path = os.path.join("/".join(notebook_param_s3_bucket_prefix.split("/")[3:]), base_local_file_path)
    download_from_s3(s3_client, local_file_path, bucket_name, s3_file_path)
    with open(local_file_path, "r") as f:
        data = f.read()
        params = json.loads(data)
    return params


# helper function to load XGBoost model into xgboost.Booster
def load_model(model_data_s3_uri):
    model_file = "./xgboost-model.tar.gz"
    bucket, key = model_data_s3_uri.replace("s3://", "").split("/", 1)
    boto3.client("s3").download_file(bucket, key, model_file)
    
    with tarfile.open(model_file, "r:gz") as t:
        t.extractall(path=".")
    
    # Load model
    model = xgb.Booster()
    model.load_model("xgboost-model")

    return model

def plot_roc_curve(fpr, tpr):
    fn = "roc-curve.png"
    fig = plt.figure(figsize=(6, 4))
    
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(fn)

    return fn

def evaluate(region,
             bucket_prefix,
             mlflow_tracking_server_arn,
             experiment_name,
             run_id=None):
    
    os.environ["AWS_DEFAULT_REGION"] = region
    boto_session = boto3.Session(region_name=region)
    sess = sagemaker.Session(boto_session=boto_session)
    bucket_name = sess.default_bucket()
    preprocess_step_name = "02-preprocess"
    train_step_name = "03-train"
    notebook_param_s3_bucket_prefix = f"s3://{bucket_name}/{bucket_prefix}/params"
    s3_client = boto3.client("s3", region_name=region)
    preprocess_step_params = read_params(s3_client, notebook_param_s3_bucket_prefix, preprocess_step_name)
    train_step_params = read_params(s3_client, notebook_param_s3_bucket_prefix, train_step_name)
    
    test_x_data_s3_path = preprocess_step_params["test_x_data"]
    test_y_data_s3_path = preprocess_step_params["test_y_data"]
    model_s3_path = train_step_params["model_s3_path"]
    output_s3_prefix = f"s3://{bucket_name}/{bucket_prefix}"
    tracking_server_arn = mlflow_tracking_server_arn
    experiment_name = preprocess_step_params["experiment_name"]

    suffix = strftime('%d-%H-%M-%S', gmtime())
    mlflow.set_tracking_uri(tracking_server_arn)
    experiment = mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"evaluate-{suffix}", nested=True)

    X_test = xgb.DMatrix(pd.read_csv(test_x_data_s3_path, header=None).values)
    y_test = pd.read_csv(test_y_data_s3_path, header=None).to_numpy()

    # Run predictions
    probability = load_model(model_s3_path).predict(X_test)
    
    # Evaluate predictions
    fpr, tpr, thresholds = roc_curve(y_test, probability)
    auc_score = auc(fpr, tpr)
    eval_result = {"evaluation_result": {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
        },
    }}
    
    mlflow.log_metric("auc_score", auc_score)
    mlflow.log_artifact(plot_roc_curve(fpr, tpr))
    prediction_baseline_s3_path = f"{output_s3_prefix}/prediction_baseline/prediction_baseline.csv"
    
    # Save prediction baseline file - we need it later for the model quality monitoring
    pd.DataFrame({"prediction":np.array(np.round(probability), dtype=int),
                  "probability":probability,
                  "label":y_test.squeeze()}
                ).to_csv(prediction_baseline_s3_path, index=False, header=True)
    
    evaluation_result = {
        **eval_result,
        "prediction_baseline_data":prediction_baseline_s3_path,
        "experiment_name":experiment.name
    }
    
    evaluation_result_json_file = "evaluation.json"

    with open(evaluation_result_json_file, "w") as f:
        f.write(json.dumps(evaluation_result))

    evaluation_result_s3_path = f"{output_s3_prefix}/evaluation/{evaluation_result_json_file}"
    s3_file_path = "/".join(evaluation_result_s3_path.split("/")[3:])
    upload_to_s3(s3_client, evaluation_result_json_file, bucket_name, s3_file_path)
    
    params = {}
    params["evaluation_result_s3_path"] = evaluation_result_s3_path

    step_name = "04-evaluation"
    write_params(s3_client, step_name, params, notebook_param_s3_bucket_prefix)
    mlflow.end_run()

    return evaluation_result['evaluation_result']['classification_metrics']['auc_score']['value']

