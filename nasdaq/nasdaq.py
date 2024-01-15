import os
import nasdaqdatalink as nsdq
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

API_KEY_FILE = 'data/nasdaqapikey'
SECONDS_IN_A_DAY = 86400
FOCUS_COLUMNS = ['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']
COL_HIGH_CLOSE = 'high-close-change'
COL_CLOSE_OPEN = 'close-open-change'
FEATURES = ['Adj. Close', 'Adj. Volume', COL_HIGH_CLOSE, COL_CLOSE_OPEN]
FORECAST = {'col': 'Adj. Close', 'predict col': 'future_close', 'days': -100}
MODEL_FILE = 'data/model.pickle'

def init():
    style.use('ggplot')


def create_change_columns(df):
    df[COL_HIGH_CLOSE] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']
    df[COL_CLOSE_OPEN] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']
    return df


def append_forecast_set_to_df(df, forecast_set, current_unix_timestamp):
    if len(forecast_set) > 0:
        x = forecast_set[0]
        new_forecast_set = np.delete(forecast_set, 0)
        next_unix_timestamp = current_unix_timestamp + SECONDS_IN_A_DAY
        unix_timestamp_to_datetime = datetime.datetime.fromtimestamp(next_unix_timestamp)
        df.loc[unix_timestamp_to_datetime] = [np.nan for _ in range(len(df.columns) - 1)] + [x]
        return append_forecast_set_to_df(
            df=df, forecast_set=new_forecast_set, current_unix_timestamp=next_unix_timestamp
        )
    else:
        return df


def create_x_y(df):
    x = np.array(df.drop([FORECAST['predict col']], 1))
    x = preprocessing.scale(x)
    y = np.array(df[FORECAST['predict col']])
    return x, y


def save_model(classifier):
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(classifier, f)


def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def train(x, y):
    print(f'len(data_x): {len(x)}, len(data_y): {len(y)}')

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.8)
    classifier = LinearRegression()
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)
    print(f'accuracy: {accuracy}')
    return classifier, accuracy


def predict(df, classifier, x):
    tmp_x = x[:FORECAST['days']]
    x_lately = tmp_x[FORECAST['days']:]
    print(f'len(x_lately): {len(x_lately)}')
    forecast_set = classifier.predict(x_lately)

    last_data_row = df.iloc[-1]
    last_data_row_name = last_data_row.name
    last_date_unix_timestamp = last_data_row_name.timestamp()
    df['forecast'] = np.nan

    print(f'Length of dataframe before forecast set is appended: {len(df)}')
    print(f'Length of forecast set: {len(forecast_set)}')
    append_forecast_set_to_df(df=df, forecast_set=forecast_set, current_unix_timestamp=last_date_unix_timestamp)
    print(f'Length of dataframe after forecast set is appended: {len(df)}')
    return df


def plot_graph(df, plot_columns):
    for c in plot_columns:
        df[c].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def main():
    init()

    nsdq.read_key(API_KEY_FILE)
    df = nsdq.get('WIKI/GOOGL')
    df = df[FOCUS_COLUMNS]
    print(df.head(10))

    df = create_change_columns(df)
    # select FEATURES columns
    df = df[FEATURES]
    df.fillna(-99999, inplace=True)

    # forecast days ahead for Close column
    df[FORECAST['predict col']] = df[FORECAST['col']].shift(FORECAST['days'])
    df.dropna(inplace=True)
    x, y = create_x_y(df)

    classifier = load_model()
    if classifier is None:
        print('Model does not exist yet.')
        classifier, accuracy = train(x, y)
        print('Saving model...')
        save_model(classifier)
    else:
        print('Model already exists. Not required to train.')

    df = predict(df, classifier, x)
    print(df.tail())

    plot_graph(df, ['Adj. Close', 'forecast'])


if __name__ == '__main__':
    main()

