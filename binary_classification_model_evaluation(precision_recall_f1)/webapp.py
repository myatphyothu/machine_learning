import pandas as pd
import numpy as np
import streamlit as st
import time

labels = {
    '"1, 0"': ['1', '0'],
    '"X, Y"': ['X', 'Y'],
    '"A, B"': ['A', 'B'],
    '"apples, oranges"': ['apples', 'oranges']
}
dataset_variables = {
    'rows': {'min_value': 10, 'max_value': 10000, 'default': 100},
    'models': {'min_value': 1, 'max_value': 5, 'default': 2}
}
dataset = None


def generate_dummy_binary_datasets():
    global dataset_variables, dataset

    n = st.session_state.rows
    label_values = labels.get(st.session_state.labels, ['x', 'y'])
    models = st.session_state.models

    data_hashmap = dict()
    data_hashmap['r'] = list(np.random.choice(label_values, size=n))
    for i in range(models):
        data_hashmap[f'm{i}'] = list(np.random.choice(label_values, size=n))
    dataset = pd.DataFrame.from_dict(data_hashmap)

    #
    st.subheader(f'Dataset: {n} records')

    st.write(f'Generated dataset...')
    # st.write(tabulate.tabulate(df, headers='keys', tablefmt='grid'))
    st.dataframe(dataset)


def compute_precision_recall_f1(df, model):
    def create_value_counts_columns(df, col, values):
        counts = df[[col]].value_counts()
        for v in values:
            count_v = counts.get(v)
            df[f'{col}_count_{v}'] = count_v
        return df

    label_values = labels.get(st.session_state.labels, ['x', 'y'])
    models = st.session_state.models
    v1, v2 = label_values

    df = df.copy()

    for i in range(models):
        df[f'accuracy_{i}'] = np.where(df['r'] == df[f'm{i}'], 1, 0)

    df = create_value_counts_columns(df, 'r', label_values)
    df = create_value_counts_columns(df, model, label_values)
    df[f'{model}_{v1}_accuracy'] = np.where((df['accuracy_0'] == 1) & (df[f'{model}'] == v1), 1, 0)
    df[f'{model}_{v2}_accuracy'] = np.where((df['accuracy_0'] == 1) & (df[f'{model}'] == v2), 1, 0)

    sum_x_accuracy = int(df[[f'{model}_{v1}_accuracy']].sum())
    sum_y_accuracy = int(df[[f'{model}_{v2}_accuracy']].sum())

    df[f'{model}_{v1}_accurate'] = sum_x_accuracy
    df[f'{model}_{v1}_inaccurate'] = df[f'{model}_count_{v1}'] - df[f'{model}_{v1}_accurate']
    df[f'{model}_{v2}_accurate'] = sum_y_accuracy
    df[f'{model}_{v2}_inaccurate'] = df[f'{model}_count_{v2}'] - df[f'{model}_{v2}_accurate']
    df[f'{model}_observations_{v1}'] = df[f'{model}_{v1}_accurate'] + df[f'{model}_{v2}_inaccurate']
    df[f'{model}_observations_{v2}'] = df[f'{model}_{v2}_accurate'] + df[f'{model}_{v1}_inaccurate']

    precision_x, precision_y = f'precision_{v1}', f'precision_{v2}'
    recall_x, recall_y = f'recall_{v1}', f'recall_{v2}'
    f1_score_x, f1_score_y = f'f1_score_{v1}', f'f1_score_{v2}'
    df[precision_x] = df[f'{model}_{v1}_accurate'] / df[f'{model}_observations_{v1}']
    df[recall_x] = df[f'{model}_{v1}_accurate'] / df[f'{model}_count_{v1}']
    df[precision_y] = df[f'{model}_{v2}_accurate'] / df[f'{model}_observations_{v2}']
    df[recall_y] = df[f'{model}_{v2}_accurate'] / df[f'{model}_count_{v2}']

    df[f1_score_x] = (df[precision_x] * df[recall_x] * 2) / (df[precision_x] + df[recall_x])
    df[f1_score_y] = (df[precision_y] * df[recall_y] * 2) / (df[precision_y] + df[recall_y])

    df[f'precision({v1}+{v2})'] = (df[precision_x] + df[precision_y]) / 2
    df[f'recall({v1}+{v2})'] = (df[recall_x] + df[recall_y]) / 2
    df[f'f1_score({v1}+{v2})'] = (df[f1_score_x] + df[f1_score_y]) / 2

    select_cols = [
        f'precision_{v1}', f'recall_{v1}', f'f1_score_{v1}',
        f'precision_{v2}', f'recall_{v2}', f'f1_score_{v2}',
        f'precision({v1}+{v2})', f'recall({v1}+{v2})', f'f1_score({v1}+{v2})']

    return df[select_cols].head(1)


def render():
    global dataset

    generate_dummy_binary_datasets()
    models = st.session_state.models
    for  i in range(models):
        model = f'm{i}'
        result_df = compute_precision_recall_f1(dataset, model)
        st.subheader(f'Precision, Recall and F1 score for model {model}')
        st.dataframe(result_df)



def get_user_input():
    global dataset_variables

    with st.sidebar:
        st.radio('Binary Labels', tuple(labels.keys()), key='labels')
        st.write("Binary Classification Model Inputs")
        st.slider('Rows',
                  min_value=dataset_variables['rows']['min_value'],
                  max_value=dataset_variables['rows']['max_value'],
                  value=100,
                  key='rows',
                  help="The number of records the model will generate")
        st.slider('Models',
                  min_value=dataset_variables['models']['min_value'],
                  max_value=dataset_variables['models']['max_value'],
                  value=1,
                  key='models')

        st.button('Generate data', key='load_data', on_click=render)
        # with st.spinner("Loading..."):
        #     time.sleep(5)
        # st.success("Done!")




def main():
    get_user_input()


if __name__ == '__main__':
    main()
