import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing, model_selection, neighbors

sys.path.append('../')
sys.path.append(os.getcwd())

import ml_utils.env_settings as env_settings
from ml_utils.model import Model


args = {}
arg_keys = ['retrain', 'predict']
project_name = 'breast_cancer'
diagnosis_map = {'M': 212, 'B': 357}
diagnosis_prediction_map = {212: 'M', 357: 'B'}


def get_cli_args():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('-retrain', '--retrain', action='store_true')
    parser.add_argument('-predict', '--predict', action='store_true')

    cli_args = parser.parse_args()

    args['cli_args'] = {}
    for k in arg_keys:
        value = getattr(cli_args, k, None)
        args['cli_args'][k] = value
        print(f'{k} ==> {value}')


def load_metadata():
    global args

    with open(args['metadata path'], 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def load_dataframe(dataset_path, columns):
    global args

    df = pd.read_csv(
        filepath_or_buffer=dataset_path,
        header=None,
        names=columns,
        index_col=None,
        delimiter=',',
        skip_blank_lines=True
    )

    return df


def map_diagnosis(df):
    # changing Malignant to 4 and Benign to 2 as classification prefers numbers
    new_diagnosis_column = df['Diagnosis'].apply(lambda x: diagnosis_map[x])
    df['Diagnosis'] = new_diagnosis_column.astype('int64')
    return df


def train(retrain=False, model_info=None):
    global args

    features = args['train metadata']['features']
    labels = args['train metadata']['labels']
    test_size = args['train metadata']['test_size']

    if model_info is None or retrain is True:
        x = np.array(args['original dataset'][features])
        y = np.array(args['original dataset'][labels])
        print(x)
        print(f'len(x): {len(x)}, len(y): {len(y)}')
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size)

        clf = neighbors.KNeighborsClassifier()
        clf.fit(x_train, y_train)

        accuracy = clf.score(x_test, y_test)

        print(f'training - model accuracy: {accuracy}')
        Model.save_model(classifier=clf, accuracy=accuracy, model_name=project_name)

    else:
        print(f'Model already exists with an accuracy {model_info["accuracy"]}. Not required to train.')


def predict(data_set):
    global args

    model = args['model info']['model']
    data_sets = np.array(data_set[args['metadata']['predict']['features']])
    measures = np.array(data_sets).reshape(len(data_sets), -1)
    predictions = model.predict(measures)
    results = [diagnosis_prediction_map[x] if x in diagnosis_prediction_map else 'N/A' for x in predictions]
    print(f'prediction: {results}')

    return results


def diagnosis_summary(prediction_results):
    global args

    df = args['predict dataset']
    diagnosis_details = ['Malignant' if x == 'M' else 'Benign' if x == 'B' else 'Not certain' for x in prediction_results]
    df.insert(2, 'Diagnosis', prediction_results, True)
    df.insert(3, 'Details', diagnosis_details, True)

    summary_df = df[['ID', 'Diagnosis', 'Details']]
    print(summary_df.head(len(summary_df)))


def main():
    global args

    get_cli_args()
    env_settings.init()
    Model.init()

    args['project path'] = os.environ["MACHINE_LEARNING_PATH"]
    args['dataset path'] = f'{args["project path"]}/datasets/breast_cancer/wdbc.csv'
    args['predict dataset path'] = f'{args["project path"]}/datasets/breast_cancer/to_predict.csv'
    args['metadata path'] = f'{args["project path"]}/datasets/breast_cancer/metadata.yaml'
    args['metadata'] = load_metadata()
    args['train metadata'] = args['metadata']['train']
    args['predict metadata'] = args['metadata']['predict']
    args['original dataset'] = map_diagnosis(
        load_dataframe(
            dataset_path=args['dataset path'],
            columns=args['train metadata']['columns']
        )
    )
    args['predict dataset'] = load_dataframe(
        dataset_path=args['predict dataset path'],
        columns=args['predict metadata']['columns']
    )
    args['model info'] = Model.load_model(project_name)
    train(retrain=args['cli_args']['retrain'], model_info=args['model info'])

    # ====================== PREDICTION ======================
    if args['cli_args']['predict'] is True:
        results = predict(args['predict dataset'])
        diagnosis_summary(results)


if __name__ == '__main__':
    main()
