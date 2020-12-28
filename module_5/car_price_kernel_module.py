import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, RandomizedSearchCV, cross_val_score
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.base import clone
from tqdm.notebook import tqdm
from pprint import pprint
import json

import matplotlib.pyplot as plt
import seaborn as sns

VERSION = 15
RANDOM_SEED = 42
VAL_SIZE = 0.20
ITERATIONS = 5000  # для CatBoost
LR = 0.1  # для CatBoost


def print_col_info(data):
    value_counts = data.value_counts()
    num_rows = 4
    if len(value_counts) > num_rows:
        value_counts = value_counts.iloc[:num_rows]
    display(value_counts)
    print(f"Num of missing values: {data.isna().sum()}")


def print_duplicates_info(data):
    print(f"Num of duplicates by car_url: {data.duplicated(subset=['car_url']).sum()}")
    print(f"Num of duplicates by sell_id: {data.duplicated(subset=['sell_id']).sum()}")
    print(f"Num of duplicates by car_url, sell_id: {data.duplicated(subset=['car_url', 'sell_id']).sum()}")


body_types = \
    ['седан',
     'внедорожник',
     'лифтбек',
     'хэтчбек',
     'универсал',
     'минивэн',
     'купе',
     'компактвэн',
     'пикап двойная кабина',
     'купе-хардтоп',
     'родстер',
     'фургон',
     'кабриолет',
     'седан-хардтоп',
     'микровэн',
     'лимузин',
     'пикап одинарная кабина',
     'пикап полуторная кабина',
     'внедорожник открытый',
     'тарга',
     'фастбек'
     ]


def clear_body_type(elem, types=body_types):
    elem = elem.strip().lower()
    appr_types = []

    for t in types:
        if t in elem:
            appr_types.append(t)

    if len(appr_types) == 0:
        print('Err1 in clear_body_type()')
        return 'other'
    else:
        max_len = 0
        max_type = ''
        for t in appr_types:
            if len(t) > max_len:
                max_len = len(t)
                max_type = t
        else:
            return max_type


def clear_engine_disp(engine_disp):
    engine_disp = engine_disp.replace('LTR', '').strip()
    if engine_disp == '':
        print('Error_1 in clear_engine_disp()')
        engine_disp = '2.0'
    return engine_disp


def clear_engine_power(engine_power):
    engine_power = engine_power.replace('N12', '').strip()
    engine_power = float(engine_power)
    return engine_power


car_drives = ['передний', 'задний', 'полный']


def clear_car_drives(drive, car_drives=car_drives):
    for cd in car_drives:
        if cd in drive:
            return cd
    else:
        return 'other'


def clear_owners(owners):
    owners = owners.strip().lower()
    return int(owners[0])


def clear_model_name(data):
    data = data.str.strip()
    data = data.str.upper()
    data = data.str.replace(' ', '_')
    data = data.str.replace('-', '_')
    data = data.str.replace('KLASSE', 'CLASS')
    data = data.str.replace('КЛАСС', 'CLASS')
    data = data.str.replace('СЕРИИ', 'SERIES')
    data = data.str.replace('RAV4', 'RAV_4')

    data = data.str.replace('RS6', 'RS_6')
    data = data.str.replace('RSQ3', 'RS_Q3')
    data = data.str.replace('RS7', 'RS_7')
    data = data.str.replace('RS4', 'RS_4')
    data = data.str.replace('RS3', 'RS_3')
    data = data.str.replace('RS5', 'RS_5')

    data = data.str.replace('DELICA_D:5', 'DELICA_D5')
    data = data.str.replace('DELICA_D_5', 'DELICA_D5')
    data = data.str.replace('DELICA_D:2', 'DELICA_D2')
    data = data.str.replace('QASHQAI\\+2', 'QASHQAI_PLUS_2')
    data = data.str.replace('\\(NORTH_AMERICA\\)', 'NA')
    data = data.str.replace("'", '')
    data = data.str.replace(":", '')
    data = data.str.replace('Z3M', 'Z3_M')
    data = data.str.replace('Z3М', 'Z3_M')

    data = data.str.replace('V40_CC', 'V40_CROSS_COUNTRY')
    data = data.str.replace('^140$', '140_SERIES')
    data = data.str.replace('S_CLASS_MAYBACH', 'MAYBACH_S_CLASS')
    data = data.str.replace('GLE_CLASS_COUPE_AMG', 'GLE_COUPE_AMG')
    data = data.str.replace('PRIUS_V_\(\+\)', 'PRIUS_PLUS')
    data = data.str.replace('PRIUSPLUS', 'PRIUS_PLUS')
    data = data.str.replace('MASTERACE_SURF', 'MASTER_ACE_SURF')
    data = data.str.replace('STEPWGN', 'STEPWAGON')
    data = data.str.replace('AMG_GLC_COUPE', 'GLC_COUPE_AMG')

    data = data.str.replace('^GLS_AMG$', 'GLS_CLASS_AMG')
    data = data.str.replace('^GLB_AMG$', 'GLB_CLASS_AMG')
    data = data.str.replace('^GLA_AMG$', 'GLA_CLASS_AMG')
    data = data.str.replace('^GLE_AMG$', 'GLE_CLASS_AMG')
    data = data.str.replace('^GLC_AMG$', 'GLC_CLASS_AMG')

    data = data.str.replace('^GLS_COUPE$', 'GLS_CLASS_COUPE')
    data = data.str.replace('^GLA_COUPE$', 'GLA_CLASS_COUPE')
    data = data.str.replace('^GLE_COUPE$', 'GLE_CLASS_COUPE')
    data = data.str.replace('^GLC_COUPE$', 'GLC_CLASS_COUPE')
    data = data.str.replace('^GLB_COUPE$', 'GLB_CLASS_COUPE')

    return data


equipment_all_categories = set()


def clear_equipment_dict(equipment_str):
    equipment_str = equipment_str.strip()
    if len(equipment_str) == 0:
        equipment_all_categories.add('other')
        return {'other': 1}
    equipment_str = equipment_str.replace('True', '1') \
        .replace('False', '0') \
        .replace('\'', '\"')
    equipment_dict = json.loads(equipment_str)
    equipment_dict_fix_keys = {k.strip().replace(' ', '_') \
                                   .replace('-', '_') \
                                   .replace(':', '_'): equipment_dict[k] for k in equipment_dict.keys()}
    for k in equipment_dict_fix_keys.keys():
        equipment_all_categories.add(k)
    return equipment_dict_fix_keys


def get_round_price(prices, round_factor=100):
    return round_factor * (np.round(prices / round_factor))


def make_submission(sample_submission, predict_submission, is_make=False, VERSION=VERSION):
    sample_submission['price'] = predict_submission
    if (is_make):
        sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)
    display(sample_submission.head(10))
    return sample_submission


def predict_by_all_data(model, X, y, X_sub):
    model.fit(X, y)
    return model.predict(X_sub)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def mape_exp(y_true, y_pred):
    y_true = np.exp(y_true)
    y_pred = np.exp(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true))


def print_mape(val):
    print(f"Точность модели по метрике MAPE: {val * 100:0.2f}%")


def get_scorer():
    return make_scorer(mape_exp, greater_is_better=False)


def mape_single(y_true, y_pred):
    return np.abs((y_pred - y_true) / y_true)


def cv_score(model, X, y):
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_val_score(model, X, y, cv=kf, n_jobs=-1, scoring=get_scorer())
    return {'cv_score_mean': np.mean(cv_results), 'cv_scores': cv_results}


def train_test_score(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True,
                                                        random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {'train_test_score': mape(y_test, y_pred)}


def get_extr_model_default(random_seed=RANDOM_SEED):
    return ExtraTreesRegressor(random_state=random_seed, n_jobs=-1)


def get_rf_model_default(random_seed=RANDOM_SEED):
    return RandomForestRegressor(random_state=random_seed, n_jobs=-1)


def get_adab_model_default(random_seed=RANDOM_SEED):
    return AdaBoostRegressor(random_state=random_seed)


def get_gradb_model_default(random_seed=RANDOM_SEED):
    return GradientBoostingRegressor(random_state=random_seed)


def get_catboost_model_default(random_seed=RANDOM_SEED):
    return CatBoostRegressor(iterations=ITERATIONS,
                             learning_rate=LR,
                             random_seed=random_seed,
                             eval_metric='MAPE',
                             custom_metric=['R2', 'MAE'],
                             # silent=True,
                             logging_level='Silent'
                             )


def get_best_model(model, param_grid, X, y):
    gs_cv = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
    gs_cv.fit(X, y)
    return gs_cv.best_estimator_, gs_cv.best_params_


def delete_not_matching_categories(train_col, test_col):
    train_vals = train_col.value_counts().index
    test_vals = test_col.value_counts().index
    not_matching_vals_1 = set(train_vals) - set(test_vals)
    not_matching_vals_2 = set(test_vals) - set(train_vals)
    print('Before: ', {'len1': len(not_matching_vals_1), 'len2': len(not_matching_vals_2)})

    train_col = train_col.apply(lambda s: 'OTHER' if (s in not_matching_vals_1 or s in not_matching_vals_2) else s)
    test_col = test_col.apply(lambda s: 'OTHER' if (s in not_matching_vals_1 or s in not_matching_vals_2) else s)

    train_vals = train_col.value_counts().index
    test_vals = test_col.value_counts().index
    not_matching_vals_1 = set(train_vals) - set(test_vals)
    not_matching_vals_2 = set(test_vals) - set(train_vals)
    print('After: ', {'len1': len(not_matching_vals_1), 'len2': len(not_matching_vals_2)})

    return train_col, test_col


def check_ndarray(x):
    if str(type(x)) != "<class 'numpy.ndarray'>":
        return x.values
    else:
        return x


def compute_meta_feature(model, X_train, X_test, y_train, cv):
    X_meta_train = np.zeros(len(y_train), dtype=np.float32)
    splits = cv.split(X_train)
    for train_fold_index, predict_fold_index in splits:
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        folded_model = clone(model)
        folded_model.fit(X_fold_train, y_fold_train)
        X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)

    if X_test.shape[0] != 0:
        meta_model = clone(model)
        meta_model.fit(X_train, y_train)
        X_meta_test = meta_model.predict(X_test)
    else:
        X_meta_test = None

    return X_meta_train, X_meta_test


def generate_meta_features(models, X_train, X_test, y_train, cv):
    X_train = check_ndarray(X_train)
    X_test = check_ndarray(X_test)
    y_train = check_ndarray(y_train)
    num_models = len(models)

    features = [
        compute_meta_feature(model, X_train, X_test, y_train, cv)
        for model in tqdm(models)
    ]

    stacked_features_train = np.hstack([
        features_train for features_train, features_test in features
    ]).reshape(-1, num_models)

    stacked_features_test = np.hstack([
        features_test for features_train, features_test in features
    ]).reshape(-1, num_models)

    return stacked_features_train, stacked_features_test


def compute_metric(model, X_train, y_train, X_test, y_test, useLog=False):
    X_train = check_ndarray(X_train)
    X_test = check_ndarray(X_test)
    y_train = check_ndarray(y_train)
    y_test = check_ndarray(y_test)

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    if y_test.shape[0] != 0:
        if useLog:
            y_test = np.exp(y_test)
            y_test_pred = np.exp(y_test_pred)
        print(f"Точность модели по метрике MAPE: {(mape(y_test, y_test_pred)) * 100:0.2f}%")

    return y_test_pred


def get_models_diff_random_state(model, num_models=5):
    init_params = model.get_params()
    init_random = init_params['random_state']
    model_list = [clone(model)]

    for n in np.arange(1, num_models):
        curr_params = init_params.copy()
        curr_params['random_state'] = init_random * (n + 1)
        curr_model = clone(model)
        curr_model = curr_model.set_params(**curr_params)
        model_list.append(curr_model)

    return model_list


def plot_cat_col_vs_price(data, col, ylim, num_categories=4):
    values = data[col].value_counts().index
    if len(values) > num_categories:
        values = values[:num_categories]
    values = list(values)
    data_curr_categories = data.query(f"{col} in {values}")
    curr_categories = break_long_names(data_curr_categories[col].values)

    plt.figure()
    sns.boxplot(x=curr_categories, y=data_curr_categories['price'])
    ylim = sorted(ylim)
    plt.ylim(ylim[0], ylim[1])
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('price')
    plt.grid()
    plt.show()


def plot_num_col_vs_price_unified(train, test, col):
    fig, axs = plt.subplots(2, 3, figsize=(15, 4))

    data = train

    sns.distplot(data[col], ax=axs[0, 0])
    axs[0, 0].set_title('Train')
    axs[0, 0].grid(True)

    sns.distplot(np.log(data[col] + 1), ax=axs[0, 1])
    axs[0, 1].set_title('Train, logarithmic')
    axs[0, 1].grid(True)

    sns.scatterplot(data=data, x=col, y='price', ax=axs[0, 2])
    axs[0, 2].set_title('Train, scatter')
    axs[0, 2].grid(True)

    data = test

    sns.distplot(data[col], ax=axs[1, 0])
    axs[1, 0].set_title('Test')
    axs[1, 0].grid(True)

    sns.distplot(np.log(data[col] + 1), ax=axs[1, 1])
    axs[1, 1].set_title('Test, logarithmic')
    axs[1, 1].grid(True)

    axs[1, 2].set_axis_off()
    fig.set_figheight(9)
    fig.set_figwidth(18)

    plt.show()


def plot_num_col_vs_price(data, col):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    sns.distplot(data[col], ax=axs[0])
    axs[0].set_title('')
    axs[0].grid(True)

    sns.distplot(np.log(data[col] + 1), ax=axs[1])
    axs[1].set_title('Logarithmic')
    axs[1].grid(True)

    sns.scatterplot(data=data, x=col, y='price', ax=axs[2])
    axs[2].set_title('Scatter')
    axs[2].grid(True)

    fig.suptitle('Train')
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


def plot_corr_matrix(data, columns):
    plt.figure(figsize=(20, 14))
    sns.heatmap(data[columns].corr().abs(), vmin=0, vmax=1, annot=True)
    plt.show()


def plot_top_categories(train, test, col, num_categories=4):
    train_value_counts = train[col].value_counts()
    train_num_missing = train[col].isna().sum()
    if len(train_value_counts) > num_categories:
        train_value_counts = train_value_counts.iloc[:num_categories]

    test_value_counts = test[col].value_counts()
    test_num_missing = test[col].isna().sum()
    if len(test_value_counts) > num_categories:
        test_value_counts = test_value_counts.iloc[:num_categories]

    train_values = break_long_names(train_value_counts.index)
    test_values = break_long_names(test_value_counts.index)

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(x=train_value_counts.values, y=train_values, ax=axs[0])
    axs[0].grid(True)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    if train_num_missing == 0:
        axs[0].set_title('Train')
    else:
        axs[0].set_title(f'Train, {train_num_missing} missing')

    sns.barplot(x=test_value_counts.values, y=test_values, ax=axs[1])
    axs[1].grid(True)
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    if test_num_missing == 0:
        axs[1].set_title('Test')
    else:
        axs[1].set_title(f'Test, {test_num_missing} missing')

    fig.suptitle(col)
    plt.show()


def break_long_names(names):
    len_name = 12
    short_names = []
    for name in names:
        if type(name) != type('') or len(name) < len_name:
            short_names.append(name)
        else:
            short_names.append(name[:len_name] + '\n' + name[len_name:])
    else:
        return short_names


def plot_feature_importances(model, data):
    feature_imp = pd.Series(model.feature_importances_, index=data.columns)
    feature_imp.sort_values(ascending=False, inplace=True)
    feature_imp = feature_imp.iloc[:15]

    plt.figure()
    sns.barplot(x=feature_imp.values, y=feature_imp.index)
    plt.title('Feature importances')
    plt.grid()
    plt.show()

