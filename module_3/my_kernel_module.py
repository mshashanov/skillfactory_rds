# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


def get_mode(col):
    return col.mode().iloc[0]


def print_isnan_percent(col):
    val = col.isna().sum() * 100 / col.shape[0]
    print(f"Процент пропусков: {round(val, 1)}")


def add_isnan_col(df, col_name):
    col_name_notnan = col_name + '_not_NAN'
    df[col_name_notnan] = (~pd.isna(df[col_name])).astype('uint8')


def add_outlier_col(df, col_name, val_range):
    val_range = sorted(val_range)
    col_name_outlier = col_name + '_not_outlier'
    df[col_name_outlier] = (df[col_name] >= val_range[0]) & (df[col_name] <= val_range[1]).astype('uint8')


def clean_num_of_reviews(df):
    col_name = 'num_of_reviews'
    add_isnan_col(df, col_name)
    mean_num_on_city = df.groupby(['city'])[col_name].mean()
    city_list = df['city'].value_counts().index
    # for city in city_list:
    # val = int(round(mean_num_on_city[city]))
    # df_output.loc[df_output['city'] == city, col_name] = df_output.loc[df_output['city'] == city, col_name].fillna(val)
    df[col_name].fillna(2, inplace=True)


def clean_cuisine_style(df):
    col_name = 'cuisine_style'
    add_isnan_col(df, col_name)
    df[col_name].fillna("['Other']", inplace=True)


def clean_price_range(df):
    col_name = 'price_range'
    add_isnan_col(df, col_name)
    # est_id_modes = df.groupby(['rest_id'])[col_name].aggregate(lambda price: price.mode() if len(price.mode()) == 1 else price.mode()[0])
    # df.loc[:,col_name] = df.apply(lambda line: line[col_name] if not pd.isna(line[col_name]) else rest_id_modes.loc[line['rest_id']], axis=1)
    mode = get_mode(df[col_name])
    df[col_name].fillna(mode, inplace=True)


def list_unpack(list_of_lists):
    result = []
    for lst in list_of_lists:
        result.extend(lst)
    return result


def preproc_cuisine_style(df):
    col_name = 'cuisine_style'
    df.loc[:, col_name] = df[col_name].str.findall(r"'(\b.*?\b)'")
    df['num_' + col_name] = df[col_name].apply(lambda style: len(style)).astype('uint8')
    cuisine_counter = Counter(list_unpack(df[col_name].tolist()))
    cuisine_list = []
    cuisine_list.extend(list(cuisine_counter.most_common(10)))
    cuisine_list.extend(list(cuisine_counter.most_common()[-10:]))
    # cuisine_list = list(cuisine_counter.most_common())
    for cuisine in cuisine_list:
        if cuisine != 'Other':
            feature_name = 'cuisine_' + cuisine[0].replace(' ', '_')
            # feature_name = cuisine
            df[feature_name] = df[col_name].apply(lambda x: 1 if cuisine in x else 0).astype('uint8')
    list_of_unique_cuisine = [x[0] for x in cuisine_counter.most_common()[-16:]]
    df['unique_cuisine_style'] = df['cuisine_style'].apply(
        lambda x: 1 if len(set(x) & set(list_of_unique_cuisine)) > 0 else 0).astype('uint8')
    list_of_comm_cuisine = [x[0] for x in cuisine_counter.most_common(16)]
    df['comm_cuisine_style'] = df['cuisine_style'].apply(
        lambda x: 1 if len(set(x) & set(list_of_comm_cuisine)) > 0 else 0).astype('uint8')


def plot_big_cities_hist(df, col_name, num_cities=10):
    city_list = (df['city'].value_counts())[:num_cities].index
    for x in city_list:
        df[col_name][df['city'] == x].hist(bins=100)
    plt.show()


def preproc_ranking_2(df):
    mean_Ranking_on_City = df.groupby(['city'])['ranking'].mean()
    count_Restorant_in_City = df['city'].value_counts(ascending=False)
    col_mean = df['city'].apply(lambda x: mean_Ranking_on_City[x])
    max_Ranking_on_City = df.groupby(['city'])['ranking'].max()
    col_max = df['city'].apply(lambda x: max_Ranking_on_City[x])
    # df['norm_Ranking_on_maxRank_in_City'] = (df['ranking'] - df['mean_Ranking_on_City']) / df['max_Ranking_on_City']
    df['norm_Ranking_on_maxRank_in_City'] = (df['ranking']) / col_max


def preproc_ranking(df):
    city_list = df['city'].value_counts().index
    scaler = MinMaxScaler()
    for city in city_list:
        df.loc[df['city'] == city, 'ranking'] = scaler.fit_transform(
            df['ranking'][df['city'] == city].values.reshape(-1, 1))


def preproc_num_of_reviews(df):
    city_list = df['city'].value_counts().index
    scaler = MinMaxScaler()
    for city in city_list:
        df.loc[df['city'] == city, 'num_of_reviews'] = scaler.fit_transform(
            df['num_of_reviews'][df['city'] == city].values.reshape(-1, 1))


def preproc_price_range(df):
    col_name = 'price_range'
    price_value_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}
    df.loc[:, col_name] = df[col_name].map(price_value_dict, na_action='ignore').astype('uint8')


def preproc_city(df):
    col_name = 'city'
    city_list = df[col_name].value_counts(ascending=False).index
    df.loc[:, col_name] = df[col_name].apply(lambda c: c if c in city_list else 'Other')
    return pd.get_dummies(df, columns=['city'], dummy_na=False)


def get_percentiles(col):
    '''Функция вычисления перцентилей и межквартильного размаха для данной колонки DataFrame'''
    list_quant = [col.quantile(0.25), col.quantile(0.75)]
    iqr = list_quant[1] - list_quant[0]
    diap = [list_quant[0] - 1.5 * iqr, list_quant[1] + 1.5 * iqr]
    return [list_quant, diap, iqr]


def print_iqr_range(col):
    print(f"Дипазон по IQR: {[int(round(elem)) for elem in get_percentiles(col)[1]]}")


def print_column_hist(col, list_borders, list_borders_plot=None):
    '''Функция вывода гистограммы и оценки выбросов для данной колонки DataFrame'''
    list_borders = sorted(list_borders)
    display(col.describe())
    list_perc = get_percentiles(col)
    print(f'25-й перцентиль: {list_perc[0][0]},', f'75-й перцентиль: {list_perc[0][1]}',
          f'\nIQR: {list_perc[-1]},', f'Границы выбросов: [{list_perc[1][0]}, {list_perc[1][1]}].')
    # col.hist(bins=8, range=list_borders, label='IQR')
    if list_borders_plot == None:
        list_borders_plot = (list_borders[0] - 2, list_borders[1] + 2)
    else:
        list_borders_plot = sorted(list_borders_plot)
    col.loc[col.between(list_borders[0], list_borders[1])].hist(alpha=0.5,
                                                                bins=100,
                                                                range=list_borders_plot,
                                                                label='Здравый смысл')
    col.loc[col.between(list_perc[1][0], list_perc[1][1])].hist(bins=100,
                                                                range=list_borders_plot,
                                                                label='IQR')
    plt.legend()


def print_unique_rest_info(df):
    nunique_dict = {}
    sep = ','
    nunique_dict['Кол-во строк'] = df.shape[0]
    nunique_dict['Кол-во уникальных по rest_id'] = df['rest_id'].nunique()
    nunique_dict['Кол-во уникальных по id_ta'] = df['id_ta'].nunique()
    nunique_dict['Кол-во уникальных по rest_id,city'] = df.apply(lambda line: line['rest_id'] + sep + line['city'],
                                                                 axis=1).nunique()
    nunique_dict['Кол-во уникальных по id_ta,city'] = df.apply(lambda line: line['id_ta'] + sep + line['city'],
                                                               axis=1).nunique()
    nunique_dict['Кол-во уникальных по rest_id,city,id_ta'] = df.apply(
        lambda line: line['rest_id'] + sep + line['city'] + sep + line['id_ta'], axis=1).nunique()
    display(nunique_dict)


def print_id_ta_duplicates_info(df, add_print=False):
    nunique_list = []
    id_ta_value_counts = df['id_ta'].value_counts()
    display(id_ta_value_counts[:10])
    num_duplicates = (id_ta_value_counts > 1).sum()
    print(f"Кол-во дубликатов по id_ta: {num_duplicates}")
    for n in range(num_duplicates):
        curr_id = id_ta_value_counts.index[n]
        rep_data = df.query("id_ta==@curr_id")
        if add_print:
            display(f'Повтор номер {n + 1}')
            display(rep_data)
        nunique_list.append(rep_data.nunique(axis=0).values.tolist())
    nunique_list = np.array(nunique_list)
    display("Макс уникальных", nunique_list.max(axis=0).tolist())
    display("Мин уникальных", nunique_list.min(axis=0).tolist())


def drop_id_ta_duplicates(df):
    id_ta_value_counts = df['id_ta'].value_counts()
    num_duplicates = (id_ta_value_counts > 1).sum()
    for n in range(num_duplicates):
        curr_id = id_ta_value_counts.index[n]
        rep_data = df.query("id_ta==@curr_id")
        new_ranking = int(round(rep_data['ranking'].mean()))
        if rep_data.shape[0] > 1:
            # df.loc[rep_data.index[0], 'ranking'] = new_ranking
            df.drop(index=rep_data.index[1], inplace=True)


def align_id_ta_duplicates(df):
    id_ta_value_counts = df['id_ta'].value_counts()
    num_duplicates = (id_ta_value_counts > 1).sum()
    for n in range(num_duplicates):
        curr_id = id_ta_value_counts.index[n]
        rep_data = df.query("id_ta==@curr_id")
        rep_data_train = rep_data.query("sample==1")
        rep_data_test = rep_data.query("sample==0")
        if rep_data.shape[0] > 1:
            df.loc[rep_data_train.index[0], 'ranking'] = df.loc[rep_data_test.index[0], 'ranking']
            df.loc[rep_data_train.index[0], 'num_of_reviews'] = df.loc[rep_data_test.index[0], 'num_of_reviews']



