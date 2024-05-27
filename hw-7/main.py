import click
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import bisect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
import pickle


@click.command()
@click.argument('command', nargs=1)
@click.option('--data', required=True, help='Training data path.')
@click.option('--test', required=False, help='Test data path.')
@click.option('--split', required=False, type=float, help='Fraction of data to be used as a test dataset.')
@click.option('--random_state', required=False, default=42, help='Random state to be used for train test split (ignored if split not spicified).')
@click.option('--model', required=True, help='Path to the model pickle file.')
def parser(command, data, test, split, random_state, model):
    """Programm that trains a model on the given data or makes a prediction."""
    match command:
        case 'train':
            data_df = get_norm_df(data)
            if test:
                test_df = get_norm_df(test)
                train_df = data_df
            elif split:
                train_df, test_df = train_test_split(data_df, test_size=split, random_state=random_state)
            else:
                train_df = data_df
                test_df = None
            train_model(train_df, test_df, model)
        case 'predict':
            predict_model(data, model)
        case _:
            pass


def get_norm_df(df_path: str) -> pd.DataFrame:
    try:
        init_df = pd.read_csv(df_path)
    except FileNotFoundError:
        print(f"Error: provided file '{df_path}' does not exist")
        exit(1)
    df = pd.DataFrame()
    try:
        df['bin_rating'] = init_df['rating'].apply(lambda x: bisect.bisect_left([3], x))
        df['text'] = init_df['text'].apply(preprocess_text)
    except KeyError:
        print("Error: incorrect data format")
        exit(1)
    return df


def preprocess_text(text: str) -> str:
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    remove_extra_symb: list[str] = re.sub(r'[^\w^\s]+', '', text).lower().split()
    return ' '.join([stemmer.stem(w) for w in remove_extra_symb if w not in stop_words])


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, model_path: str):
    bow = CountVectorizer()
    x_train = bow.fit_transform(train_df['text'])
    y_train = train_df['bin_rating']

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_train, y_train)
    try:
        with open(model_path, 'wb') as f:
            pickle.dump((log_reg, bow), f)
    except:
        print(f"Error: could not save model to file '{model_path}'")
        exit(1)

    if test_df is not None:
        x_test = bow.transform(test_df['text'])
        y_test = test_df['bin_rating']
        y_pred = log_reg.predict(x_test)
        print(f"Model got f1_score {f1_score(y_pred, y_test, average='weighted')}")


def predict_model(data: str, model_path):
    try:
        with open(model_path, 'rb') as f:
            log_reg: LogisticRegression
            bow: CountVectorizer
            log_reg, bow = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: file '{model_path}' does not exist")
        exit(1)

    if os.path.exists(data):
        df = get_norm_df(data)
        try:
            x = bow.transform(df['text'])
        except KeyError:
            print("Error: incorrect data format")
            exit(1)
    else:
        x = bow.transform([preprocess_text(data)])

    preds = log_reg.predict(x)
    for pred in preds:
        print("Negative" if pred == 0 else "Positive")


if __name__ == '__main__':
    parser()
