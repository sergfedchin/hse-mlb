import pytest
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
import re

DATA_PATH = '../data/singapore_airlines_reviews.csv'
SIZE = 250
SPLIT = 0.2
RANDOM_STATE = 68

@pytest.fixture
def clear_workdir():
    for filename in os.listdir():
        if filename.split('.')[-1] in ['csv', 'pkl']:
            print(f"os.remove({filename})")
            os.remove(filename)


@pytest.fixture
def make_small_dataset():
    df = pd.read_csv(DATA_PATH)
    small_df = df.sample(n=SIZE)
    data = f'test_df_{SIZE}.csv'
    small_df.to_csv(data)
    yield data
    os.remove(data)


@pytest.fixture
def get_model():
    model = 'test_model.pkl'
    yield model
    os.remove(model)


def test_train_simple_ok(clear_workdir, get_model, make_small_dataset):
    model, data = get_model, make_small_dataset
    print("OK")
    assert os.WIFEXITED(os.system(f'python3 main.py train --data "{data}" --model "{model}"'))
    with open(model, 'rb') as f:
        log_reg, bow = pickle.load(f)
    assert isinstance(log_reg, LogisticRegression) and isinstance(bow, CountVectorizer)


@pytest.fixture
def fixture_train_bad_data():
    df = pd.read_csv(DATA_PATH)
    data = 'bad_df.csv'
    df.sample(n=SIZE).drop(columns='text').to_csv(data)
    yield 'no_model.pkl', data
    os.remove(data)


def test_train_bad_data(fixture_train_bad_data):
    model, data = fixture_train_bad_data
    assert os.WEXITSTATUS(os.system(f'python3 main.py train --data "{data}" --model "{model}"')) == 1


def test_train_no_data():
    model, data = 'no_model.pkl', 'no_data.csv'
    assert os.WEXITSTATUS(os.system(f'python3 main.py train --data "{data}" --model "{model}"')) == 1


@pytest.fixture
def get_output_file():
    output = "out.txt"
    yield output
    os.remove(output)


def test_predict_ok(get_model, make_small_dataset, get_output_file):
    model, data = get_model, make_small_dataset
    output_filename = get_output_file
    assert os.WIFEXITED(os.system(f'python3 main.py train --data "{data}" --model "{model}" --split {SPLIT} --random_state {RANDOM_STATE} > {output_filename}'))
    assert os.WIFEXITED(os.system(f'python3 main.py predict --data "{data}" --model "{model}" > {output_filename}'))
    assert os.WIFEXITED(os.system(f'python3 main.py predict --data "Good airline! Fast high-level service." --model "{model}" >> {output_filename}'))
    with open(output_filename, 'rb') as f:
        output = f.read()
    assert set(output.split()).issubset({b'Positive', b'Negative'}) and len(output.split()) == SIZE + 1


@pytest.fixture
def make_train_test_dataset():
    df = pd.read_csv(DATA_PATH)
    full_size = int(SIZE / (1 - SPLIT))
    small_df = df.sample(n=full_size)
    data_train = f'test_df_{full_size}_train.csv'
    data_test = f'test_df_{full_size}_test.csv'
    df_train, df_test = train_test_split(small_df, test_size=SPLIT, random_state=RANDOM_STATE)
    df_train.to_csv(data_train)
    df_test.to_csv(data_test)
    yield data_train, data_test
    os.remove(data_train)
    os.remove(data_test)


def find_floats(string: str) -> list[float]:
    floating_points = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string)
    return [float(num) for num in floating_points]


def test_train_with_test(get_model, make_train_test_dataset, get_output_file):
    model = get_model
    data_train, data_test = make_train_test_dataset
    output_filename = get_output_file
    assert os.WIFEXITED(os.system(f'python3 main.py train --data "{data_train}" --test "{data_test}" --model "{model}" > "{output_filename}"'))
    with open(model, 'rb') as f:
        log_reg, bow = pickle.load(f)
    assert isinstance(log_reg, LogisticRegression) and isinstance(bow, CountVectorizer)
    with open(output_filename, 'r') as f:
        output = f.read()
    metric = find_floats(output)
    assert len(metric) == 2 and 0 <= metric[-1] and metric[-1] <= 1


@pytest.fixture
def make_train_test_orig_dataset():
    df = pd.read_csv(DATA_PATH)
    full_size = int(SIZE / (1 - SPLIT))
    small_df = df.sample(n=full_size)
    data_train = f'test_df_{full_size}_train.csv'
    data_test = f'test_df_{full_size}_test.csv'
    data = f'test_df_{full_size}.csv'
    df_train, df_test = train_test_split(small_df, test_size=SPLIT, random_state=RANDOM_STATE)
    df_train.to_csv(data_train)
    df_test.to_csv(data_test)
    small_df.to_csv(data)
    yield data_train, data_test, data
    os.remove(data_train)
    os.remove(data_test)
    os.remove(data)

# test if we will get the same result by manually splitting the dataset and
# specifying random state and split size
def test_train_split(get_model, make_train_test_orig_dataset, get_output_file):
    model = get_model
    data_train, data_test, data = make_train_test_orig_dataset
    output_filename = get_output_file
    assert os.WIFEXITED(os.system(f'python3 main.py train --data "{data_train}" --test "{data_test}" --model "{model}" > "{output_filename}"'))
    assert os.WIFEXITED(os.system(f'python3 main.py train --data "{data}" --split {SPLIT} --random_state {RANDOM_STATE} --model "{model}" >> "{output_filename}"'))
    with open(output_filename, 'r') as f:
        output = f.read()
    metric = find_floats(output)
    assert len(metric) == 4 and metric[1] == metric[3]