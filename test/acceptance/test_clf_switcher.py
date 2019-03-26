import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import re as re
from models.clf_switcher import CLfSwitcher


@pytest.fixture
def dataset():
    data_train = pd.read_csv('../data/interim/train.csv')
    data_train.set_index('PassengerId', inplace=True)
    y_train = data_train.pop('Survived')
    X_train = data_train
    X_test = pd.read_csv('../data/interim/test.csv')
    X_test.set_index('PassengerId', inplace=True)
    y_test = pd.read_csv('../data/interim/gender_submission.csv')
    y_test.set_index('PassengerId', inplace=True)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def pipeline(dataset):
    X_train, X_test, y_train, y_test = dataset
    # select_numeric_cols = FunctionTransformer(lambda X: X.select_dtypes(exclude=['object']), validate=False)
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # inf = SimpleImputer(missing_values=np.inf, strategy='mean')
    return Pipeline([
        # ('select_numeric', clean_dataset([X_train, X_test])),
        # ('select_imp', imp),
        # ('select_inf', inf),
        ('clf', CLfSwitcher())
    ])


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def clean_dataset(full_data):
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['FamilySize2'] = dataset['FamilySize'] ** 2

        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

        dataset['Fare'] = dataset['Fare'].fillna(full_data[0]['Fare'].median())

        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()

        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
        dataset['Title'] = dataset['Name'].apply(get_title)
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # Mapping Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']

    train = full_data[0].drop(drop_elements, axis=1)


    test = full_data[1].drop(drop_elements, axis=1)

    return train, test


def test_hyper_parameter_optimizer(pipeline, dataset):
    X_train, X_test, y_train, y_test = dataset

    parameters = [
        {'clf__estimator': [SVC()],
         'clf__estimator__C': [0.1, 0.5, 0.9],
         'clf__estimator__kernel': ['linear', 'rbf', 'sigmoid']
         }
    ]

    gscv = GridSearchCV(
        pipeline, parameters, cv=5, n_jobs=12, verbose=3,
    )
    X_train, X_test = clean_dataset([X_train, X_test])
    # param optimization
    gscv.fit(X_train, y_train)
    score = gscv.score(X_test, y_test)
    print(f'MY SCORE: {score}')
    assert isinstance(score, float)
