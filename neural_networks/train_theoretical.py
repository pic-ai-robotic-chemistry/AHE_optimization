# -*- coding: utf-8 -*-
"""
:File: train_theoretical.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import os
import sys
import time

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(sys.argv[0]))

data = 'data_theoretical.xlsx'
x = pd.read_excel(data, sheet_name='metals')
y = pd.read_excel(data, sheet_name='params')
seed = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
norm = StandardScaler().fit(y_train)
y_train_ = norm.transform(y_train)
y_test_ = norm.transform(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(y.shape[1])
])
model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
model.fit(x_train, y_train_, batch_size=256, epochs=1000, validation_data=(x_test, y_test_), verbose=2, callbacks=[
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_delta=1e-5, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, patience=15)
])

file = f'model_theoretical_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
model.save(file)
joblib.dump(norm, f'{file}/norm.pkl')
