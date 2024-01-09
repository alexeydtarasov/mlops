import subprocess
import datetime

import pandas as pd

import utils


def infer():
    dataset = utils.get_dataset()
    model = utils.load_model()
    model_params = dict()
    y_pred, metrics = utils.get_model_metrics(dataset, model, model_params, True)

    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv('results/predictions.csv')

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subprocess.run(['dvc', 'add', 'results/predictions.csv'])
    subprocess.run(['dvc', 'push'])

    subprocess.run(['git', 'add', 'results/'])
    subprocess.run(['git', 'commit', '-m', f'{now}: prediction was uploaded to dvc'])
    subprocess.run(['git', 'push'])


if __name__ == '__main__':
    infer()
