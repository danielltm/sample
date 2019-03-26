from src.transform import get_standar_scale_with_pca_etc
import pytest
from sklearn.datasets import load_boston
from sklearn.linear_model import  Ridge
from sklearn.model_selection import train_test_split
import pandas as pd

@pytest.fixture
def dataset():
    data = load_boston()
    return (
        pd.DataFrame(data['data'], columns=data.feature_names),
        pd.Series(data['target'], name='PRICE')
    )

@pytest.fixture
def sample(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def test_transform_etl_with_ridge(sample):
    pipe = get_standar_scale_with_pca_etc()
    pipe.steps.append(('regressor', Ridge()))

    X_train, X_test, y_train, y_test = sample

    pipe = pipe.fit(X_train,y_train)
    score = pipe.score(X_test, y_test)
    assert isinstance(score, float)
    assert score > 0