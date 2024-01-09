import os

from sklearn.ensemble import RandomForestRegressor

from adtarasov_mlops import classes, utils


def fit_model(
    dataset: classes.Dataset, model_params: dict = dict()
) -> RandomForestRegressor:
    model = RandomForestRegressor(**model_params)
    model.fit(dataset.X_train, dataset.y_train)

    return model


def train(dataset_path: str, models_path: str, models_dump_name: str):
    print("Getting dataset")
    dataset = utils.get_dataset(dataset_path)
    print("Got dataset")
    model_params = dict()
    print("Fitting model")
    model = fit_model(dataset, model_params)
    print("Evaluating model")
    model_metrics = utils.get_model_metrics(dataset, model, model_params)
    print(f'Model metrics: {model_metrics}')
    print("Saving model")
    utils.save_model(model, os.path.join(models_path, models_dump_name))
