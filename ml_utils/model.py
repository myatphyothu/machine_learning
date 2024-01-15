import os
import sys
import pickle
import yaml

sys.path.append('../')
sys.path.append(os.getcwd())

import ml_utils.numpy_dtypes as numpy_dtypes

class Model(object):

    models = {}
    data_path = None

    @staticmethod
    def init():
        Model.data_path = f"{os.environ['MACHINE_LEARNING_PATH']}/models/models.yaml"
        if os.path.exists(Model.data_path):
            with open(Model.data_path, 'r') as yaml_file:
                Model.models = yaml.safe_load(yaml_file)

    @staticmethod
    def load_model(model_name):
        if model_name in Model.models:
            model_path = Model.models[model_name]['file']
            accuracy = Model.models[model_name]['accuracy']
            if os.path.exists(model_path):
                with open(model_path, 'rb') as pickle_file:
                    model = pickle.load(pickle_file)
                    return {'model': model, 'accuracy': accuracy}
            else:
                return None
        else:
            print(f'Model - {model_name} - does not exist yet.')
            return None

    @staticmethod
    def save_model(classifier, accuracy, model_name):
        model_path = f"{os.environ['MACHINE_LEARNING_PATH']}/models/{model_name}.pickle"
        Model.models[model_name] = {}
        Model.models[model_name]['file'] = model_path
        Model.models[model_name]['accuracy'] = numpy_dtypes.to_python_primitive_type(accuracy)
        with open(model_path, 'wb') as pickle_file:
            pickle.dump(classifier, pickle_file)

        print(Model.models)
        with open(Model.data_path, 'w') as yaml_file:
            yaml.safe_dump(Model.models, yaml_file)
            print(f'New model - {model_name} - was saved to {model_path}.')
