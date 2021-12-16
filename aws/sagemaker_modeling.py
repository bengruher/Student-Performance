import argparse
import joblib
import os
import json
from io import StringIO
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=7)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=50)
    
    args = parser.parse_args()
    
    # Take the set of files and read them all into a single pandas dataframe
    train_dir = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(train_dir) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
        
    input_files = list()
    for item in train_dir:
        if os.path.isdir(item): # item in train folder may be a directory with files (directory can't be read by Pandas)
            subitems = os.listdir(item)
            for file in subitems:
                input_files.append(item + '/' + file) # add full path to filename
        else:
            input_files.append(item)
    print('Input files: ', input_files)
    raw_data = [ pd.read_csv(file) for file in input_files ]
    concat_data = pd.concat(raw_data)
        
    # separate features and target variable
    X = concat_data[concat_data.columns[1:]]
    y = concat_data[concat_data.columns[0]]
    
    hyperparameters = {
        "max_depth": args.max_depth,
        "verbose": 1,  # show all logs
        "min_samples_leaf": args.min_samples_leaf,
        "min_samples_split": args.min_samples_split,
        "n_estimators": args.n_estimators,
    }
    
    print("Training the regressor")
    model = RandomForestRegressor()
    model.set_params(**hyperparameters)
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """
    Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
