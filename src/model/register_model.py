# # # register model

# # import json
# # import mlflow
# # import logging
# # from src.logger import logging
# # import os
# # import dagshub

# # import warnings
# # warnings.simplefilter("ignore", UserWarning)
# # warnings.filterwarnings("ignore")

# # # Below code block is for production use
# # # -------------------------------------------------------------------------------------
# # # # Set up DagsHub credentials for MLflow tracking
# # # dagshub_token = os.getenv("CAPSTONE_TEST")
# # # if not dagshub_token:
# # #     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# # # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# # # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# # # dagshub_url = "https://dagshub.com"
# # # repo_owner = "aryan-Patel-web"
# # # repo_name = "production-sentiment-analysis-end-to-end-mlops"
# # # # Set up MLflow tracking URI
# # # mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# # # -------------------------------------------------------------------------------------


# # # Below code block is for local use
# # # -------------------------------------------------------------------------------------
# # mlflow.set_tracking_uri('https://dagshub.com/aryan-Patel-web/production-sentiment-analysis-end-to-end-mlops.mlflow')
# # dagshub.init(repo_owner='aryan-Patel-web', repo_name='production-sentiment-analysis-end-to-end-mlops', mlflow=True)
# # # -------------------------------------------------------------------------------------


# # def load_model_info(file_path: str) -> dict:
# #     """Load the model info from a JSON file."""
# #     try:
# #         with open(file_path, 'r') as file:
# #             model_info = json.load(file)
# #         logging.debug('Model info loaded from %s', file_path)
# #         return model_info
# #     except FileNotFoundError:
# #         logging.error('File not found: %s', file_path)
# #         raise
# #     except Exception as e:
# #         logging.error('Unexpected error occurred while loading the model info: %s', e)
# #         raise

# # def register_model(model_name: str, model_info: dict):
# #     """Register the model to the MLflow Model Registry."""
# #     try:
# #         model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
# #         # Register the model
# #         model_version = mlflow.register_model(model_uri, model_name)
        
# #         # Transition the model to "Staging" stage
# #         client = mlflow.tracking.MlflowClient()
# #         client.transition_model_version_stage(
# #             name=model_name,
# #             version=model_version.version,
# #             stage="Staging"
# #         )
        
# #         logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
# #     except Exception as e:
# #         logging.error('Error during model registration: %s', e)
# #         raise

# # def main():
# #     try:
# #         model_info_path = 'reports/experiment_info.json'
# #         model_info = load_model_info(model_info_path)
        
# #         model_name = "my_model"
# #         register_model(model_name, model_info)
# #     except Exception as e:
# #         logging.error('Failed to complete the model registration process: %s', e)
# #         print(f"Error: {e}")

# # if __name__ == '__main__':
# #     main()


# import os
# import json
# import joblib
# import mlflow
# import logging
# import dagshub
# import traceback
# import shutil

# # -------------------- LOGGING --------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # -------------------- DAGSHUB --------------------
# mlflow.set_tracking_uri(
#     "https://dagshub.com/aryan-Patel-web/production-sentiment-analysis-end-to-end-mlops.mlflow"
# )

# dagshub.init(
#     repo_owner='aryan-Patel-web',
#     repo_name='production-sentiment-analysis-end-to-end-mlops',
#     mlflow=True
# )

# # -------------------- SAVE MODEL INFO --------------------
# def save_model_info(run_id, model_path, file_path):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     info = {
#         "run_id": run_id,
#         "model_path": model_path
#     }

#     with open(file_path, "w") as f:
#         json.dump(info, f, indent=4)

#     logging.info("✅ experiment_info.json saved")


# # -------------------- MAIN --------------------
# def main():
#     try:
#         logging.info("🚀 STARTING MODEL EVALUATION")

#         mlflow.set_experiment("my-dvc-pipeline")

#         # ---------- LOAD MODEL ----------
#         model_path = "models/model.pkl"

#         if not os.path.exists(model_path):
#             raise FileNotFoundError("❌ model.pkl not found")

#         model = joblib.load(model_path)

#         logging.info(f"✅ Model loaded: {type(model)}")

#         # ---------- START RUN ----------
#         with mlflow.start_run() as run:

#             logging.info(f"🔥 RUN_ID: {run.info.run_id}")

#             # ---------- METRICS ----------
#             accuracy = 1.0
#             mlflow.log_metric("accuracy", accuracy)

#             # SAVE metrics.json for DVC
#             os.makedirs("reports", exist_ok=True)

#             with open("reports/metrics.json", "w") as f:
#                 json.dump({"accuracy": accuracy}, f, indent=4)

#             logging.info("✅ metrics.json saved")

#             # ---------- FORCE MODEL SAVE ----------
#             if os.path.exists("temp_model"):
#                 shutil.rmtree("temp_model")

#             os.makedirs("temp_model", exist_ok=True)

#             # Save model locally
#             joblib.dump(model, "temp_model/model.pkl")

#             # Log folder as artifact
#             mlflow.log_artifacts("temp_model", artifact_path="model")

#             logging.info("✅ Model FORCE logged as artifact")

#             # ---------- VERIFY ----------
#             client = mlflow.tracking.MlflowClient()
#             artifacts = client.list_artifacts(run.info.run_id)

#             logging.info("📂 ARTIFACTS FOUND:")
#             for art in artifacts:
#                 logging.info(f" - {art.path}")

#             # ---------- SAVE RUN INFO ----------
#             save_model_info(
#                 run.info.run_id,
#                 "model",
#                 "reports/experiment_info.json"
#             )

#             logging.info("🎉 DONE SUCCESSFULLY")

#     except Exception as e:
#         logging.error("❌ ERROR")
#         logging.error(str(e))
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()

import json
import mlflow
import mlflow.sklearn
import logging
import os
import dagshub
import warnings
from src.logger import logging

warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')


# Below code block is for production use
# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "aryan-Patel-web"
repo_name = "production-sentiment-analysis-end-to-end-mlops"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/aryan-Patel-web/production-sentiment-analysis-end-to-end-mlops.mlflow')
# dagshub.init(repo_owner='aryan-Patel-web', repo_name='production-sentiment-analysis-end-to-end-mlops', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model run ID and path from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Register the model to the MLflow Model Registry and transition to Staging."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info('Registering model from URI: %s', model_uri)

        # Register the model in MLflow Model Registry
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info('Model registered: %s, version: %s', model_name, model_version.version)

        # Transition the model version to "Staging"
        # client = mlflow.tracking.MlflowClient()
        # client.transition_model_version_stage(
        #     name=model_name,
        #     version=model_version.version,
        #     stage='Staging'
        # )

        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage='Staging',
        archive_existing_versions=True
        )

        logging.info(
            'Model %s version %s transitioned to Staging',
            model_name,
            model_version.version
        )
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = 'my_model'
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()