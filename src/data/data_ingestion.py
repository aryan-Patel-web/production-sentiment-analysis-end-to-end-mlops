# # data ingestion
# import numpy as np
# import pandas as pd
# pd.set_option('future.no_silent_downcasting', True)

# import os
# from sklearn.model_selection import train_test_split
# import yaml

# import logging

# from src.logger import logging

# from src.connections import s3_connection

# from dotenv import load_dotenv

# load_dotenv()

# bucket_name = os.getenv("S3_BUCKET_NAME")
# accesskey = os.getenv("AWS_ACCESS_KEY_ID")
# secretkey = os.getenv("AWS_SECRET_ACCESS_KEY")




# def load_params(params_path: str) -> dict:
#     """Load parameters from a YAML file."""
#     try:
#         with open(params_path, 'r') as file:
#             params = yaml.safe_load(file)
#         logging.debug('Parameters retrieved from %s', params_path)
#         return params
#     except FileNotFoundError:
#         logging.error('File not found: %s', params_path)
#         raise
#     except yaml.YAMLError as e:
#         logging.error('YAML error: %s', e)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error: %s', e)
#         raise

# def load_data(data_url: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(data_url)

#         logging.info('Data loaded from %s', data_url)
#         return df
#     except pd.errors.ParserError as e:
#         logging.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     """Preprocess the data."""
#     try:
#         # df.drop(columns=['tweet_id'], inplace=True)
#         logging.info("pre-processing...")

#         final_df = df[df['sentiment'].isin(['positive', 'negative'])]
#         final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
#         logging.info('Data preprocessing completed')
#         return final_df
#     except KeyError as e:
#         logging.error('Missing column in the dataframe: %s', e)
#         raise
#     except Exception as e:
#         logging.error('Unexpected error during preprocessing: %s', e)
#         raise

# def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
#     """Save the train and test datasets."""
#     try:
#         raw_data_path = os.path.join(data_path, 'raw')
#         os.makedirs(raw_data_path, exist_ok=True)

#         train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
#         test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

#         logging.debug('Train and test data saved to %s', raw_data_path)
#     except Exception as e:
#         logging.error('Unexpected error occurred while saving the data: %s', e)
#         raise

# def main():
#     try:
#         params = load_params(params_path='params.yaml')
#         test_size = params['data_ingestion']['test_size']
#         # test_size = 0.2
    
        
#         # df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
#         # df = load_data(data_url='https://raw.githubusercontent.com/aryan-Patel-web/production-sentiment-analysis-end-to-end-mlops/main/notebooks/data.csv')

    
#         s3 = s3_connection.s3_operations(bucket_name, accesskey, secretkey)
#         df = s3.fetch_file_from_s3("data.csv")



#         final_df = preprocess_data(df)
#         train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

#         save_data(train_data, test_data, data_path='./data')

#     except Exception as e:
#         logging.error('Failed to complete the data ingestion process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()

# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml

import logging

from src.logger import logging

from src.connections import s3_connection

from dotenv import load_dotenv

load_dotenv()

# ✅ FIXED: correct env variable name
bucket_name = os.getenv("AWS_BUCKET_NAME")
accesskey = os.getenv("AWS_ACCESS_KEY_ID")
secretkey = os.getenv("AWS_SECRET_ACCESS_KEY")

# ✅ safety check (VERY IMPORTANT)
if not bucket_name:
    raise ValueError("❌ AWS_BUCKET_NAME is not set")



def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)

        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # df.drop(columns=['tweet_id'], inplace=True)
        logging.info("pre-processing...")

        # ✅ safety check to avoid None crash
        if df is None:
            raise ValueError("❌ Dataframe is None. S3 fetch failed.")

        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info('Data preprocessing completed')
        return final_df

    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2

        # df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        # df = load_data(data_url='https://raw.githubusercontent.com/aryan-Patel-web/production-sentiment-analysis-end-to-end-mlops/main/notebooks/data.csv')

        # ✅ DEBUG print (remove later if needed)
        print("🔥 DEBUG BUCKET:", bucket_name)

        # ✅ Using env-based AWS auth (clean approach)
        s3 = s3_connection.s3_operations(bucket_name)

        df = s3.fetch_file_from_s3("data.csv")

        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        save_data(train_data, test_data, data_path='./data')

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()