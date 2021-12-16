import argparse
import joblib
import os
import json
import sys
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sagemaker_containers import _encoders as encoders
from sagemaker_containers import _worker as worker

# the following may be required in order to get the SKLearn container to deploy properly
# see the following issue: https://github.com/aws/sagemaker-python-sdk/issues/648
# module_path = os.path.abspath('/opt/ml/code')
# if module_path not in sys.path:
#     sys.path.append(module_path)

feature_columns_names = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2']

label_column = 'G3'

numeric_cols_to_keep = ['age', 'Medu', 'traveltime', 'studytime', 'failures', 'goout', 'Dalc', 'absences']
nominal_cols_to_keep = ['address', 'Fjob', 'guardian', 'higher', 'internet', 'romantic']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(
        file, 
        header=0, 
        index_col=0) for file in input_files ]
    concat_data = pd.concat(raw_data)
        
    preprocessor = ColumnTransformer(
         transformers = [("numeric", MinMaxScaler(), numeric_cols_to_keep),
                         ("nominal", OneHotEncoder(drop='if_binary', handle_unknown='error'), nominal_cols_to_keep)],
         remainder = 'drop',
         n_jobs = -1
    )
    
    preprocessor.fit(concat_data)
    
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))
    
    
def input_fn(input_data, content_type):
    """Parse input data payload

    This function is used by Amazon Sagemaker only during inference.
    We will only allow text/csv format. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV
        # We need to use StringIO because the input_data will be the actual csv data, not the filename
        df = pd.read_csv(StringIO(input_data), index_col=0)
        
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the G3 label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names
        
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
    
    
def output_fn(prediction, accept):
    """Format prediction output

    This function is used by Amazon Sagemaker only during inference.
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), accept, mimetype=accept) # we use the Sagemaker container helper classes to return the proper types
#         return json.dumps(json_output)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept) # we use the Sagemaker container helper classes to return the proper types
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))
        
        
def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    """
    features = model.transform(input_data)
    
    # if labels were passed in, we need to add them back to the dataset because ColumnTransformer will remove them
    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features
    
def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
