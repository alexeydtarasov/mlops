import joblib
import subprocess
import datetime
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import classes


def save_model(model: RandomForestRegressor, path: str = 'models/rf.joblib'):
    joblib.dump(model, path)


def load_model(path: str = 'models/rf.joblib') -> RandomForestRegressor:
    return joblib.load(path)


def get_dataset(
    path: str = 'data/insurance.csv', test_size: float = 0.2
) -> classes.Dataset:
    # loading dataset
    subprocess.call(['dvc', 'pull', '--force', '--with-deps', 'data/insurance.csv'])
    df = pd.read_csv(path)
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['region'] = df['region'].map(
        {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    )
    x = df.drop(columns=['charges'])
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    return classes.Dataset(X_train, y_train, X_test, y_test)


def get_model_metrics(
    dataset: classes.Dataset, model: RandomForestRegressor, model_params: dict
) -> classes.Metrics:
    y_pred = model.predict(dataset.X_test)
    return classes.Metrics(
        model_params,
        r2=r2_score(dataset.y_test, y_pred),
        mse=mean_squared_error(dataset.y_test, y_pred),
        mae=mean_absolute_error(dataset.y_test, y_pred),
    )


def push_model(model_path: str = 'models/'):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subprocess.run(["dvc", "add", model_path, 'rf.joblib'])
    subprocess.run(["dvc", "push"])
    subprocess.run(["git", "add", model_path, 'rf.joblib'])
    subprocess.run(["git", "commit", "-m", f"'{now} train model added'"])
    subprocess.run(['git', 'push'])