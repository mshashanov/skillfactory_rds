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
    engine_disp = engine_disp.replace('LTR','').strip()
    if engine_disp == '':
        engine_disp = '2.0'
    return engine_disp
    
    
    
def clear_engine_power(engine_power):
    engine_power = engine_power.replace('N12','').strip()
    engine_power = float(engine_power)
    return engine_power


car_drives = ['передний', 'задний', 'полный']

def clear_car_drives(drive, car_drives=car_drives):
    for cd in car_drives:
        if cd in drive:
            return cd
    else:
        return 'other'   
    
    
    