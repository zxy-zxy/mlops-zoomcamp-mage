if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


import mlflow
import mlflow.sklearn
import os
import pickle

@data_exporter
def export_data(*args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    lr, dv = args[0]
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Start an MLflow run
    with mlflow.start_run():
        # Log the model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        
        # Save the DictVectorizer artifact
        dv_path = "dict_vectorizer.pkl"
        with open(dv_path, "wb") as f:
            pickle.dump(dv, f)
        
        # Log the artifact
        mlflow.log_artifact(dv_path, artifact_path="dict_vectorizer")

        # Clean up the local artifact file
        os.remove(dv_path)
