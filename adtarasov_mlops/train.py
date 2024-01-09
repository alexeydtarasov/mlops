import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import classes
import utils


def fit_model(
    dataset: classes.Dataset, model_params: dict = dict()
) -> RandomForestRegressor:
    model = RandomForestRegressor(**model_params)
    model.fit(dataset.X_train, dataset.y_train)

    return model


def train(data_path: str = None):
    print("Getting dataset")
    dataset = utils.get_dataset()
    print("Got dataset")
    model_params = dict()
    print("Fitting model")
    model = fit_model(dataset, model_params)
    print("Evaluating model")
    model_metrics = utils.get_model_metrics(dataset, model, model_params)
    print(model_metrics)
    print("Saving model")
    utils.save_model(model)
    utils.push_model()


if __name__ == '__main__':
    train()
