# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns 



def get_percentiles(col):
    '''Функция вычисления перцентилей и межквартильного размаха для данной колонки DataFrame'''
    list_quant = [col.quantile(0.25), col.quantile(0.75)]
    iqr = list_quant[1] - list_quant[0]
    diap = [list_quant[0] - 1.5 * iqr, list_quant[1] + 1.5 * iqr]
    return [list_quant, diap, iqr]


def print_iqr_range(col):
    print(f"Дипазон по IQR: {[int(round(elem)) for elem in get_percentiles(col)[1]]}")


def print_column_hist_old(col, list_borders, list_borders_plot=None):
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
    

def print_column_hist(col, list_borders_plot=None):
    '''Функция вывода гистограммы и оценки выбросов для данной колонки DataFrame'''
    list_borders = [col.min(), col.max()]
    display(col.describe())
    list_perc = get_percentiles(col)
    print(f'25-й перцентиль: {list_perc[0][0]},', f'75-й перцентиль: {list_perc[0][1]}',
          f'\nIQR: {list_perc[-1]},', f'Границы выбросов: [{list_perc[1][0]}, {list_perc[1][1]}].')

    if list_borders_plot == None:
        list_borders_plot = (round(list_borders[0] - 2), round(list_borders[1] + 2))
    else:
        list_borders_plot = sorted(list_borders_plot)
    
    col.hist(alpha=0.5,
        bins=100,
        range=list_borders_plot,
        color='tab:red',
        label='Original')
    col.loc[col.between(list_perc[1][0], list_perc[1][1])].hist(bins=100,
        range=list_borders_plot,
        color='tab:blue',                                                        
        label='IQR')
    plt.legend()

    
def plot_regions_hist(df, col_name, hist_range):
    regions_list = sorted(df['region_rating'].unique())
    
    #fig, axes = plt.subplots(2, 4, figsize=(25,12))
    # for region in regions_list:
    for region, i in zip(regions_list, range(7)):    
        data = df[col_name][df['region_rating'] == region]
        display(region, data.describe())
        #sns.distplot(data, kde=False, ax=axes.flat[i])
        # df[col_name][df['region_rating'] == region].hist(bins=100, range=hist_range)
           
    # plt.show()
 

    
    
    
    