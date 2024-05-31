import sys
import re
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack
import numpy as np


MODEL_PATH = 'model.pkl'
INT_TO_LANG = {0: 'BASH',
               1: 'PYTHON',
               2: 'KOTLIN',
               3: 'JAVA',
               4: 'JAVASCRIPT',
               5: 'C',
               6: 'HASKELL',
               7: 'MARKDOWN',
               8: 'YAML',
               9: 'CPP',
               10: 'OTHER'}


def get_features_puctuation(text: str):
    features = []
    text = str(text)
    for sign in string.punctuation:
        features.append(text.count(sign))
    return features


def remove_punctuation(text: str):
    return re.sub(r"[^\w\s]+", ' ', text).replace('_', ' ')


def count_uppercase_letters(text: str):
    return sum(1 for char in text if char.isupper())


def camel_case_split(text: str):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text.replace('\n', ' '))
    return ' '.join([m.group(0) for m in matches])


def get_text_features(text):
    text = str(text)
    punct = get_features_puctuation(text)
    punct.append(count_uppercase_letters(text))
    return punct


def preprocess_text(text):
    text = str(text)
    return ' '.join(camel_case_split(remove_punctuation(text)).lower().split())


def make_model_input(bow: CountVectorizer, text: str):
    features = get_text_features(text)
    bow_features = bow.transform([preprocess_text(text)])
    text_features = csr_matrix(features)
    model_input = hstack((bow_features, text_features))
    return model_input


def make_prediction(model, model_input):
    p = model.predict(model_input)
    return INT_TO_LANG[p[0]]


if __name__ == '__main__':
    """
    Для запуска тестов сделана обёртка run_tests.py, которая запускает
    этот файл со всеми тестами.
    """
    try:
        filepath = sys.argv[1]
    except IndexError:
        raise Exception("Usage: main.py <path_to_txt>")

    with open(MODEL_PATH, 'rb') as model_binary:
        model: LogisticRegression
        bow: CountVectorizer
        model, bow = pickle.load(model_binary)

    with open(filepath, 'r') as f:
        text = f.read()

    prediction = make_prediction(model=model, model_input=make_model_input(bow, text))
    print(f'{filepath}:', prediction)
