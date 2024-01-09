import datetime
import os
import subprocess

import pandas as pd

from adtarasov_mlops import utils


def infer(
    dataset_path: str,
    models_path: str,
    models_dump_name: str,
    predictions_path: str,
    metrics_path: str,
):
    print("Loading dataset")
    dataset = utils.get_dataset(dataset_path)
    print("Loading model")
    model = utils.load_model(os.path.join(models_path, models_dump_name))
    model_params = dict()
    print('Calculating model inference')
    y_pred, metrics = utils.get_model_metrics(dataset, model, model_params, True)
    print(f'Model metrics: {metrics}')
    print("Saving predictions and metrics")
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(predictions_path)
    with open(metrics_path, 'w') as fout:
        fout.write(metrics.to_json())

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subprocess.run(['dvc', 'add', predictions_path])
    subprocess.run(['dvc', 'add', metrics_path])
    subprocess.run(['dvc', 'push'])

    subprocess.run(['git', 'add', os.path.basename(predictions_path)])
    subprocess.run(['git', 'commit', '-m', f'{now}: prediction was uploaded to dvc'])
    subprocess.run(['git', 'push'])
