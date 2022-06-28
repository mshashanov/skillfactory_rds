import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import zipfile
import csv
import cv2
import sys
import os
import re, math, random, time, gc, string, pathlib, itertools
import shutil


def clear_directory(path):
    is_exist_before = os.path.exists(path)
    if is_exist_before:
        shutil.rmtree(path)
    is_exist_after = os.path.exists(path)
    if not is_exist_before:
        print('Directory is not exist')
    if is_exist_after:
        print('Directory is exist after')


def plot_classes_hist(values):
    plt.figure()
    sns.countplot(x=values)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Classes histogram')
    plt.grid()
    plt.show()


def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    axs[0].plot(epochs, acc, 'b', label='Training acc')
    axs[0].plot(epochs, val_acc, 'g', label='Validation acc')
    axs[0].set_title('Training and validation accuracy')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(epochs, loss, 'b', label='Training loss')
    axs[1].plot(epochs, val_loss, 'g', label='Validation loss')
    axs[1].set_title('Training and validation loss')
    axs[1].legend()
    axs[1].grid()

    plt.show()


def show_first_images(generator, count=6, labels=True, figsize=(20, 5), normalized=False):
    generator = itertools.islice(generator, count)
    fig, axes = plt.subplots(nrows=1, ncols=count, figsize=figsize)
    for batch, ax in zip(generator, axes):
        if labels:
            img_batch, labels_batch = batch
            img, label = img_batch[0], np.argmax(labels_batch[0]) # берем по одному изображению из каждого батча
        else:
            img_batch = batch
            img = img_batch[0]
        if not normalized:
            img = img.astype(np.uint8)
        ax.imshow(img)
        # метод imshow принимает одно из двух:
        # - изображение в формате uint8, яркость от 0 до 255
        # - изображение в формате float, яркость от 0 до 1
        if labels:
            ax.set_title(f'Class: {label}')
    plt.grid(False)
    plt.show()
    
    
def print_col_info(data, num_dec=1, num_categories=4):
    value_counts = data.value_counts(normalize=True, ascending=False)
    if len(value_counts) > num_categories:
        value_counts = value_counts.iloc[:num_categories]
    value_counts = value_counts.multiply(100).round(num_dec)
    value_counts = value_counts.apply(lambda x: str(x) + ' %')
    display(value_counts)
    print(f"Num of missing values: {data.isna().sum()}")
    

def plot_top_categories(train, test, col, num_dec=1, num_categories=4, rotation=None):
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
    
    train_value_counts = train_value_counts.multiply(100).round(num_dec)
    test_value_counts = test_value_counts.multiply(100).round(num_dec)
    
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
    
    
def get_verified_str(col_data):
    true_val = 'True' # 'True'
    false_val = 'False' # 'False'
    verified_dict = {0: false_val, False: false_val, 'False': false_val, 1: true_val, True: true_val, 'True': true_val}
    return col_data.map(verified_dict)


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
    
    
def plot_num_col_unified(train, test, col):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4/2))
    data = train
    
    try:
        sns.distplot(data[col], ax=axs[0])
    except:
        sns.distplot(data[col], ax=axs[0], kde_kws={'bw': 0.1})
    
    axs[0].set_title('train')
    axs[0].set_xlabel('')
    axs[0].grid(True)
       
    data = test
    
    try:
        sns.distplot(data[col], ax=axs[1])
    except:
        sns.distplot(data[col], ax=axs[1], kde_kws={'bw': 0.1})
    
    axs[1].set_title('test')
    axs[1].set_xlabel('')
    axs[1].grid(True)
    
    fig.set_figheight(5)
    fig.set_figwidth(15)
    fig.suptitle(col)
    plt.show()    
    
    
def plot_learning_rate(lr_list, num_epochs):
    plt.figure()
    plt.plot(range(0,num_epochs+1), lr_list, 'b', label='Exponential Decay')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Learning Rate')
    plt.legend()
    plt.xlim(0, num_epochs)
    # plt.ylim(0, LR)
    plt.grid()
    plt.show()
    
    
    
    
    
    