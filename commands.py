import yaml
import fire

from adtarasov_mlops import train as train_module
from adtarasov_mlops import infer as infer_module


def train():
    train_module.train(
        dataset_path=config['data']['dataset_path'],
        models_path=config['models']['models_path'],
        models_dump_name=config['models']['models_dump_name'],
    )


def infer():
    infer_module.infer(
        dataset_path=config['data']['dataset_path'],
        models_path=config['models']['models_path'],
        models_dump_name=config['models']['models_dump_name'],
        predictions_path=config['results']['predictions_path'],
        metrics_path=config['results']['metrics_path'],
    )


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/config.yaml'))
    fire.Fire()
