import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import make_scorer, roc_curve
from sklearn.base import clone
from tqdm.notebook import tqdm
from pprint import pprint
import json
import sklearn
from datetime import date, datetime
import os
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
import sklearn
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
from skopt import forest_minimize
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_SEED    = 42
NUM_THREADS    = 1 # число потоков
NUM_COMPONENTS = 30 # число параметров вектора 
NUM_EPOCHS     = 20 # число эпох обучения
LR             = 0.1 # темп обучения


def get_score(rating_true, rating_pred):
    return sklearn.metrics.roc_auc_score(rating_true, rating_pred)


def get_std_model_params():
    params = {'learning_rate': LR,
              'no_components': NUM_COMPONENTS,
              'random_state': RANDOM_SEED,
              'loss': 'logistic'}
    return params


def get_std_model():
    params = get_std_model_params()
    return LightFM(**params)


def objective(params, train_data, test_data, search_params_list):
    # unpack
    learning_rate = params[0]
    curr_params = get_std_model_params()
    curr_params['learning_rate'] = learning_rate
    model = LightFM(**curr_params)

    model = model.fit(train_data.copy(),
                      epochs=NUM_EPOCHS,
                      num_threads=NUM_THREADS)

    preds = model.predict(test_data.userid.values,
                          test_data.itemid.values)

    roc_auc = get_score(test_data.rating, normalize_predictions(preds))
    search_params_list.append(np.round([learning_rate, roc_auc], 6).tolist())

    # Make negative because we want to _minimize_ objective
    out = -roc_auc
    
    return out


def get_missed_ids(data, col):
    ids = data[col].unique()
    missed_ids = []
    for id in range(ids.min(), ids.max() + 1):
        if not id in ids:
            missed_ids.append(id)
    return missed_ids


def get_columns_with_diff_types(df):
    columns_with_list = []
    columns_with_str = []
    columns_with_other = []
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        if col_data.shape[0] == 0:
            columns_with_other.append(col)
        else:
            first_val = col_data.iloc[0]
            
            if type(first_val) == type(list()):
                columns_with_list.append(col)
            elif type(first_val) == type(str()):
                columns_with_str.append(col)
            else:
                columns_with_other.append(col)
                
    return columns_with_list, columns_with_str, columns_with_other


def replace_str_list_by_joined_str(df_init):
    df = df_init.copy()
    columns_with_list, _, __ = get_columns_with_diff_types(df)
    sep = '; '
    rep_na = '-'
    
    for col in columns_with_list:
        col_data_not_na = df[col].dropna()
        first_elem = 0
        for idx in col_data_not_na.index:
            curr_elem = col_data_not_na.loc[idx]
            if type(curr_elem) == type(list()) and len(curr_elem) > 0:
                first_elem = curr_elem[0]
                break
                
        if type(first_elem) != type(str()):
            continue
        else:
            df[col] = df[col].fillna(rep_na)
            df[col] = df[col].apply(lambda str_list: sep.join(sorted(str_list)) if str_list != rep_na else rep_na)
            df[col] = df[col].apply(lambda str_list: str_list[0] if type(str_list) == type(list()) and len(str_list) == 1 else str_list)
            
    return df


def sort_dict_by_keys(unsorted_dict):
    return dict(sorted(unsorted_dict.items()))


def dict_list_to_str_list(items_list):
    if items_list == 'missed':
        return ['-']
    else:
        return [str(sort_dict_by_keys(item)) for item in items_list]

    
def dict_to_str(item):
    if item == 'missed':
        return '-'
    else:
        return str(sort_dict_by_keys(item))

    
def list_to_str(item):
    if item == 'missed':
        return '-'
    else:
        sep = '; '
        if type(item) == type(list()) and len(item) == 1:
            return item[0]
        elif type(item) == type(list()) and len(item) > 1:
            return sep.join(sorted(item))
        else:
            return item


def get_top_categories(col_data, num_categories=4):
    values = col_data.value_counts(ascending=False).index
    if len(values) > num_categories:
        values = values[:num_categories]
    
    return values.tolist()


def get_top_categories_col(col_data, num_categories=4, other_cat='Other'):
    top_categories = get_top_categories(col_data, num_categories-1)
    return col_data.apply(lambda val: val if val in top_categories else other_cat)
    

def plot_cat_col_vs_rating(df, col, num_categories=4, plot_others=True):
    df = df.copy()
    top_categories = get_top_categories(df[col], num_categories)
    df[col] = df[col].apply(lambda value: value if value in top_categories else 'Others')
    if not plot_others:
        df = df.query(f"{col} != 'Others'")
    
    plt.figure()
    ax = sns.countplot(x=col, hue='rating', data=df)
    plt.grid()
    plt.show()
    
    
def get_verified_str(col_data):
    true_val = 'True' # 'True'
    false_val = 'False' # 'False'
    verified_dict = {0: false_val, False: false_val, 'False': false_val, 1: true_val, True: true_val, 'True': true_val}
    return col_data.map(verified_dict)
    
    
def plot_cat_col_vs_rating_percent(df, col, num_categories=4, gen_category='Others', ignored_categories=[], rotation=None):
    df = df.copy()
    top_categories = get_top_categories(df[col], num_categories)
    df[col] = df[col].apply(lambda value: value if value in top_categories else gen_category)
    df = df.query(f"{col} not in {ignored_categories}")
    x, y = col, 'rating'
    value_counts = df[col].value_counts()
    df_group = df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()
    df_group['value_counts'] = df_group[col].apply(lambda value: value_counts.loc[value])
    df_group.sort_values(by=['value_counts', 'rating'], ascending=False, inplace=True)
    
    if (col in ['verified', 'has_image', 'has_feature', 'has_similar_item', 'is_amazon_customer', 'is_kindle_customer']):
        df_group[col] = get_verified_str(df_group[col])
    
#     display(df_group)
    plt.figure()
#     sns.catplot(x=x, y='percent', hue=y, kind='bar', data=df_group)

    ax = sns.barplot(x=x, y='percent', hue=y, data=df_group)
    if rotation != None:
        plt.setp(ax.get_xticklabels(), rotation=rotation)
    plt.xlabel('')
    plt.ylabel('percent')
    plt.title(col)
    plt.ylim(0, 105)
    plt.grid()
#     plt.legend(loc='upper left')
    plt.show()
    
    
def plot_num_col_vs_rating(df, col, ylim=[]):
    plt.figure()
    sns.boxplot(x=df['rating'], y=df[col])
    if len(ylim) == 2:
        ylim = sorted(ylim)
        plt.ylim(ylim[0], ylim[1])
    plt.grid()
    plt.show()
    
    
def print_percentiles_info(col_data):
    info = col_data.describe()
    info.drop(labels=['count'], inplace=True)
    info['iqr'] = info['75%'] - info['25%']
    info['out_high'] = info['75%'] + 1.5 * info['iqr']
    info['out_low'] = info['25%'] - 1.5 * info['iqr']
    info = info.round(2)
    display(info)
    

def transform_meta_data_into_str(df):
    df = df.copy()
    df['similar_item'] = df['similar_item'].fillna('missed')
    df['similar_item'] = df['similar_item'].apply(dict_list_to_str_list)
    df['tech1'] = df['tech1'].fillna('missed')
    df['tech1'] = df['tech1'].apply(dict_to_str)
    df['rank'] = df['rank'].fillna('missed')
    df['rank'] = df['rank'].apply(list_to_str)
    
    df_prep = replace_str_list_by_joined_str(df)
    columns_with_list, columns_with_str, columns_with_other = get_columns_with_diff_types(df_prep)    
    if len(columns_with_list) != 0 or len(columns_with_other) != 0:
        print('Not fully prepared')
    
    return df_prep


def check_ndarrays_inclusion(arr0, arr1):
    return np.isin(arr0, arr1).all()


def sparse_matrix_equality(m0, m1):
    if m1.get_shape() != m0.get_shape():
        return not (m0 != m1)
    else:
        return (m0 != m1).nnz == 0


def get_features_from_ids(ids, var=1):
    if var == 0:
        id_vals = np.sort(ids.unique())
        id_num = id_vals.shape[0]
        features = sparse.coo_matrix(([1] * id_num, (id_vals, id_vals)))
    else:
        id_num = 1 + ids.max()
        features = sparse.identity(id_num, format='coo')
    return features


def add_units_column(matrix):
    num_rows = matrix.copy().toarray().shape[0]
    unit_feature = np.ones((num_rows, 1), dtype=np.uint8)
    unit_feature = sparse.coo_matrix(unit_feature, dtype=np.uint8)
    return sparse.hstack([unit_feature, matrix.copy()])


def get_features_from_main_cat(data):
    data = data.copy()
    data.drop_duplicates(subset=['itemid', 'asin'], inplace=True)
    data.sort_values(by=['itemid'], inplace=True)
    itemid_values = data.itemid.values
    cols_to_drop = ['verified','reviewTime','asin','reviewerName','unixReviewTime','userid','itemid','rating']
    data.drop(columns=cols_to_drop, inplace=True)
    num_features = data.shape[1]
    full_num_items = itemid_values.max() + 1
    full_features = np.zeros((full_num_items, num_features), dtype=np.uint8)
    full_features[itemid_values.tolist(), :] = data.values
    return sparse.coo_matrix(full_features, dtype=np.uint8)


def get_items(data):
    data = data.copy()
    data.drop_duplicates(subset=['itemid', 'asin'], inplace=True)
    data.sort_values(by=['itemid'], inplace=True)
    data.drop(columns=['verified','reviewTime','asin','reviewerName','unixReviewTime','userid','rating'], inplace=True)
    return data


def get_item_features(df, asin_2_main_cat_map):
    data = df[['itemid', 'asin']].drop_duplicates()
    data.sort_values(by=['itemid'], inplace=True)
    data['main_cat'] = data['asin'].map(asin_2_main_cat_map)
    return data


def get_mean_price(price_str):
    try:
        price_str_split = price_str.split('-')
        price_list = [float(elem.strip().lower()) for elem in price_str_split]
        return str(np.mean(price_list))
    except Exception:
        print('Exception: ', price_str, type(price_str))
        return '0.0'
    
    
def prepare_meta_data(meta_data):
    meta_data = meta_data.copy()
    
    col = 'asin'
    meta_data[col] = meta_data[col].str.strip()
    meta_data[col] = meta_data[col].str.upper()
    
    col = 'brand'
    meta_data[col] = meta_data[col].fillna('Unknown')
    meta_data[col] = meta_data[col].str.strip()
    meta_data[col] = meta_data[col].str.title()
    
    col = 'main_cat'
    meta_data[col] = meta_data[col].fillna('Unknown')
    meta_data[col] = meta_data[col].str.strip()
    meta_data[col] = meta_data[col].str.title()
    
    col = 'title'
    meta_data[col] = meta_data[col].fillna('Unknown')
    meta_data[col] = meta_data[col].str.strip()
    meta_data[col] = meta_data[col].apply(lambda s: s if len(s) < 50e3 else 'Unknown')
    
    col = 'rank'
    meta_data[col] = meta_data[col].fillna('Unknown')
    meta_data[col] = meta_data[col].str.replace(',','.')
    meta_data[col] = meta_data[col].str.extract(r'(\d+[.,]?\d*)')
    meta_data[col] = meta_data[col].astype(np.float64)
    meta_data[col] = meta_data[col].fillna(0.)
    
    col = 'price'
    meta_data[col] = meta_data[col].fillna('$0.0')
    meta_data[col] = meta_data[col].str.strip()
    meta_data[col] = meta_data[col].str.lower()
    meta_data[col] = meta_data[col].str.replace('$','')
    meta_data[col] = meta_data[col].str.replace(',','')
    meta_data[col] = meta_data[col].apply(get_mean_price)
    meta_data[col] = meta_data[col].astype(np.float64)
    meta_data.loc[meta_data[col] == 0, col] = np.round(meta_data[col].mean(), 2)
    meta_data[col] = meta_data[col].round(2)
    
    return meta_data


def prepare_data(data):
    data = data.copy()
    
    col = 'asin'
    data[col] = data[col].str.strip()
    data[col] = data[col].str.upper()
    
    col = 'verified'
    data[col] = data[col].map({True: 1, False: 0}).astype(np.uint8)
    
    curr_idx = data.index[0]
    dt_format = '%m %d, %Y'
    
    col = 'reviewTime'
    if type(data.loc[curr_idx, col]) == str:
        data[col] = data[col].str.strip()
        data[col] = data[col].str.lower()
        data[col] = pd.to_datetime(data[col], format=dt_format)
    else:
        print('reviewTime isn`t processed')
    
    col = 'unixReviewTime'
    if type(data.loc[curr_idx, col]) == np.int64:
        data[col] = pd.to_datetime(data[col], unit='s')
    else:
        print('unixReviewTime isn`t processed')
        
    col = 'reviewerName'
    data[col] = data[col].fillna('Unknown')
    data[col] = data[col].str.strip()
    data[col] = data[col].str.title()
    
    col = 'rating'
    if col in data.columns:
        data[col] = data[col].astype(np.uint8)
        
    idx_type = np.int64
    
    col = 'userid'
    data[col] = data[col].astype(idx_type)
    
    col = 'itemid'
    data[col] = data[col].astype(idx_type)
        
    col = 'Id'
    if col in data.columns:
        data[col] = data[col].astype(idx_type)
        
    return data


def print_duplicates_info(data):
    print(f"Num of full duplicates: {data.duplicated().sum()}")
    print(f"Num of duplicates by userid, itemid: {data.duplicated(subset=['userid', 'itemid']).sum()}")
    col = 'Id'
    if col in data.columns:
        print(f"Num of duplicates by userid, itemid, {col}: {data.duplicated(subset=['userid', 'itemid', col]).sum()}")
        
        
def prepare_duplicates(data):
    data = data.copy()
    data_duplicated = data[data.duplicated(subset=['userid', 'itemid'], keep=False)]    
    data_duplicated.sort_values(by=['userid', 'itemid', 'rating'], inplace=True)

    elem_set = {elem for elem in zip(data_duplicated.userid, data_duplicated.itemid)}
    elem_set = sorted(elem_set)
    num_rows = 0
    idx_to_remove = []

    for userid, itemid in elem_set:
        curr_data = data_duplicated.query(f"userid=={userid} & itemid=={itemid}")
        num_rows += curr_data.shape[0]
        curr_data.sort_values(by=['unixReviewTime'], inplace=True)
        min_rating = curr_data.rating.min()
        
        if min_rating != 0:
            idx_to_remove += list(curr_data.index[:-1])
        else:
            curr_data_rating_0 = curr_data.query(f"rating==0")            
            curr_idx_to_remove = list(curr_data.index)
            curr_idx_to_remove.remove(curr_data_rating_0.index[-1])
            idx_to_remove += curr_idx_to_remove
    else:
        full_num_rows = data_duplicated.shape[0]
        check_1 = (num_rows == full_num_rows)
        check_2 = (len(idx_to_remove) == (full_num_rows - len(elem_set)))
        if not (check_1 and check_2):
            print(f"prepare_duplicates error: {full_num_rows}, {num_rows}, {len(idx_to_remove)}")
            
    return data_duplicated, idx_to_remove


def get_ratings_coo_matrix(data):
    return sparse.coo_matrix((data['rating'],
                             (data['userid'],
                              data['itemid'])))


def normalize_predictions(preds):
    preds = preds.copy()
    preds -= preds.min()
    preds /= preds.max()
    return preds


def coo_to_csr_matrix(coo_matrix):
    return sparse.csr_matrix(coo_matrix.copy().toarray(), dtype=np.float32)


def print_col_info(data, num_categories=4):
    value_counts = data.value_counts(normalize=True, ascending=False)
    if len(value_counts) > num_categories:
        value_counts = value_counts.iloc[:num_categories]
    value_counts = value_counts.multiply(100).round(1)
    value_counts = value_counts.apply(lambda x: str(x) + ' %')
    display(value_counts)
    print(f"Num of missing values: {data.isna().sum()}")
    
    
def plot_pred_dist(preds, normalized_preds):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    
    sns.distplot(preds, ax=axs[0])
    axs[0].set_title('Predictions')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].grid(True)

    sns.distplot(normalized_preds, ax=axs[1])
    axs[1].set_title('Normalized predictions')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].grid(True)
    
    plt.show()


def plot_num_col_unified(train, test, col):
    fig, axs = plt.subplots(2, 2, figsize=(15, 4))
    data = train
    
    try:
        sns.distplot(data[col], ax=axs[0,0])
    except:
        sns.distplot(data[col], ax=axs[0,0], kde_kws={'bw': 0.1})
    
    axs[0,0].set_title('train')
    axs[0,0].set_xlabel('')
    axs[0,0].grid(True)
    
    try:
        sns.distplot(np.log(data[col] + 1), ax=axs[0,1])
    except:
        sns.distplot(np.log(data[col] + 1), ax=axs[0,1], kde_kws={'bw': 0.1})
    
    axs[0,1].set_title('train, logarithmic')
    axs[0,1].set_xlabel('')
    axs[0,1].grid(True)
       
    data = test
    
    try:
        sns.distplot(data[col], ax=axs[1,0])
    except:
        sns.distplot(data[col], ax=axs[1,0], kde_kws={'bw': 0.1})
    
    axs[1,0].set_title('test')
    axs[1,0].set_xlabel('')
    axs[1,0].grid(True)
    
    try:
        sns.distplot(np.log(data[col] + 1), ax=axs[1,1])
    except:
        sns.distplot(np.log(data[col] + 1), ax=axs[1,1], kde_kws={'bw': 0.1})
    
    axs[1,1].set_title('test, logarithmic')
    axs[1,1].set_xlabel('')
    axs[1,1].grid(True)
    
#     axs[1,2].set_axis_off()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    fig.suptitle(col)
    plt.show()
    
    
def plot_num_col(data, col):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    
    sns.distplot(data[col], ax=axs[0])
    axs[0].set_title('')
    axs[0].grid(True)
    
    sns.distplot(np.log(data[col] + 1), ax=axs[1])
    axs[1].set_title('Logarithmic')
    axs[1].grid(True)
    
    if col != 'price':
        fig.suptitle('Test')
    plt.show()
    
    
def break_long_names(names):
    len_name = 25
    short_names = []
    for name in names:
        if type(name) != type('') or len(name) < len_name:
            short_names.append(name)
        else:
            sep = '&'
            if sep in name:
                short_names.append(name.replace(sep, sep+os.linesep))
            else:
                short_names.append(name[:len_name] + os.linesep + name[len_name:])
    else:
        return short_names


def plot_cat_col(data, col, num_categories=4, rotation=None):
    data = data.copy()
    if col in ['rating']:
        data[col] = get_verified_str(data[col])
    
    data_value_counts = data[col].value_counts(normalize=True, ascending=False)
    data_num_missing = data[col].isna().sum()
    if len(data_value_counts) > num_categories:
        data_value_counts = data_value_counts.iloc[:num_categories]

    data_value_counts = data_value_counts.multiply(100).round(1)
    
    fig = plt.figure()
    sns.barplot(y=data_value_counts.values, x=data_value_counts.index)
    plt.grid(True)
    plt.xlabel('')
    plt.ylabel('percent')
    if data_num_missing == 0:
        plt.title(f'{col}')
    else:
        plt.title(f'{col}, {data_num_missing} missing')
        
    if rotation != None:
        plt.setp(plt.gca().get_xticklabels(), rotation=rotation)
    
    plt.show()


def plot_top_categories(train, test, col, num_categories=4, rotation=None):
    train = train.copy()
    test = test.copy()
    if (col in ['verified', 'has_image', 'has_feature', 'has_similar_item', 'is_amazon_customer', 'is_kindle_customer']):
        train[col] = get_verified_str(train[col])
        test[col] = get_verified_str(test[col])

    train_value_counts = train[col].value_counts(normalize=True, ascending=False)
    train_num_missing = train[col].isna().sum()
    if len(train_value_counts) > num_categories:
        train_value_counts = train_value_counts.iloc[:num_categories]
    
    test_value_counts = test[col].value_counts(normalize=True, ascending=False)
    test_num_missing = test[col].isna().sum()
    if len(test_value_counts) > num_categories:
        test_value_counts = test_value_counts.iloc[:num_categories]
    
    train_value_counts = train_value_counts.multiply(100).round(1)
    test_value_counts = test_value_counts.multiply(100).round(1)
    
    train_values = break_long_names(train_value_counts.index)
    test_values = break_long_names(test_value_counts.index)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(y=train_value_counts.values, x=train_values, ax=axs[0])
    axs[0].grid(True)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('percent')
    if train_num_missing == 0:
        axs[0].set_title('train')
    else:
        axs[0].set_title(f'train, {train_num_missing} missing')
    
    sns.barplot(y=test_value_counts.values, x=test_values, ax=axs[1])
    axs[1].grid(True)
    axs[1].set_xlabel('')
    axs[1].set_ylabel('percent')
    if test_num_missing == 0:
        axs[1].set_title('test')
    else:
        axs[1].set_title(f'test, {test_num_missing} missing')
    
    fig.suptitle(col)
    if rotation != None:
        plt.setp(axs[0].get_xticklabels(), rotation=rotation)
        plt.setp(axs[1].get_xticklabels(), rotation=rotation)
    plt.show()

    
def get_cat_cols_le(data, cat_cols):
    cat_cols_le = []
    
    for col in cat_cols:
        new_col = col + '_le'
        cat_cols_le.append(new_col)
        data[new_col] = LabelEncoder().fit_transform(data[col])
        
        if data[col].nunique() != data[new_col].nunique():
            print(f"Error in column {col}")

    return cat_cols_le
    

def get_verified_percent(data):
    value_counts = data.value_counts(normalize=True, ascending=False)
        
    if 1 in value_counts.index:
        verified_percent = value_counts.loc[1]
    else:
        verified_percent = 0    
    
    return np.round(verified_percent * 100, 2)
    
    
def brand_2_verified_percent(brand, train_brand_grouped):   
    if brand in train_brand_grouped.index:
        return train_brand_grouped.loc[brand, 'verified']
    else:
        return 50.

    
def plot_corr_matrix(data, columns):
    plt.figure(figsize=(15,10)) # figsize=(21,15)
    sns.heatmap(data[columns].corr().abs(), vmin=0, vmax=1, annot=True, cmap=sns.cm.rocket)
    plt.show()
    
    
def plot_bin_cat_importance(data, columns):
    columns_for_plot = [col.replace('_le','') for col in columns]
    
    importance_values = mutual_info_classif(data[columns],
                                            data['rating'],
                                            discrete_features=True)

    bin_cat_importance = pd.Series(importance_values, index=columns_for_plot)
    bin_cat_importance.sort_values(ascending=False, inplace=True)

    plt.figure(figsize=(7.5, 4))
    sns.barplot(x=bin_cat_importance.values, y=bin_cat_importance.index)
    plt.xlabel('feature importance by mutual information')
    plt.title('Binary, categorical features')
    plt.grid()
    plt.show()
    

def plot_num_importance(data, columns):
    importance_values = f_classif(data[columns],
                                  data['rating'])[0]

    num_importance = pd.Series(importance_values, index=columns)
    num_importance.sort_values(ascending=False, inplace=True)

    plt.figure(figsize=(7.5, 4))
    sns.barplot(x=num_importance.values, y=num_importance.index)
    plt.xlabel('feature importance by F statistic')
    plt.title('Numerical features')
    plt.grid()
    plt.show()
    
    
def plot_roc_curve(rating_true, rating_pred):
    fpr, tpr, thresholds = roc_curve(rating_true, rating_pred)
    roc_auc = get_score(rating_true, rating_pred)

    plt.figure(figsize=(7.5, 4))
    plt.plot([0, 1], label='Baseline', linestyle='--')
    plt.plot(fpr, tpr, label = 'LightFM')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.title('ROC-AUC = {:7.6f}'.format(roc_auc))
    plt.ylabel('true positive rate')
    plt.xlabel('false positive rate')
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()
    
    
    
    
    
    
    
    
    


