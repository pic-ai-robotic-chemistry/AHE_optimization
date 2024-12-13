# -*- coding: utf-8 -*-
"""
:File: train_calibrated.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import time

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = 'data_calibrated.xlsx'
x = pd.read_excel(data, sheet_name='metals')
y = pd.read_excel(data, sheet_name='overpotential')
seed = 3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
norm = StandardScaler().fit(y_train)
y_train_ = norm.transform(y_train)
y_test_ = norm.transform(y_test)

premodel = tf.keras.models.load_model('model_theoretical')
premodel.trainable = False
model = tf.keras.models.Sequential([
    premodel,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
model.fit(x_train, y_train_, batch_size=2, epochs=1000, validation_data=(x_test, y_test_), verbose=2, callbacks=[
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_delta=1e-5, min_lr=1e-5),
    tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, patience=15)])

file = f'model_calibrated_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
model.save(file)
joblib.dump(norm, f'{file}/norm.pkl')
