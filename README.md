# Предсказание затрат на мед. услуги

-- В этом проекте реализовано обучение и инференс модели решающей задачу регрессии -- предсказания затрат на медицинские услуги по характеристикам человека

Задача взята с [kaggle.com](www.kaggle.com), по этой [ссылке](https://www.kaggle.com/code/shitalandhalkar/regression-medical-cost-personal/input) расположены используемые данные.

Для решения задачи используется `RandomForestRegressor` из пакета `scikit-learn`

## Проделанные этапы работы:

1. Добавлен и сконфигурирован `pyproject.yaml` посредством `poetry`
2. Добавлен и настроен `pre-commit` испольуются хуки: `black, isort, flake8`
3. Создана связка с DVC (Google Drive), по этой [ссылке](https://drive.google.com/drive/folders/1TfegdEVRJUzHZbA8Cz0Eyrui5J3molZ0?usp=share_link) расположена папка со всеми данными.
4. Конфигурация путей для обучения и получения предсказаний с помощью модели используется `yaml` конфиг


## Запуск кода

### Для запуска обучения модели:

`poetry run python commands.py train`

### Для получения предсказаний модели:

`poetry run python commmands.py infer`
