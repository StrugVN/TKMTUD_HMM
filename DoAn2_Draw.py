import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from tensorflow.keras.datasets.mnist import load_data

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import pickle
import os.path

import matplotlib.pyplot as plt

from progressbar import Percentage, ProgressBar,Bar,ETA
import time

(x_train, y_train), (x_test, y_test) = load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_train.shape, x_test.shape

def drop_const_pixel(train, to_drop = None):
    df = pd.DataFrame(train, columns = ['pixel_{0}'.format(i+1) for i in range(train.shape[1])])
    if to_drop is None:
        to_drop = []
        for c in df.columns:
            if max(df[c]) == 0 or min(df[c]) == 255:
                to_drop.append(c)
            
    df.drop(to_drop, axis = 1, inplace = True)
    
    return df.values, to_drop

def normalize(train, val):
    scalers = []
    drop_cols = []
    scaled_X_train, scaled_X_val = [], []
    
    pbar = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                       maxval=100).start()

    for i in pbar(range(len(train))):
        train_i, drop_col = drop_const_pixel(train[i])
        if val is not None:
            val_i, _ = drop_const_pixel(val[i], drop_col)
        s = StandardScaler().fit(train_i)

        scaled_X_train.append(s.transform(train_i))
        if val is not None:
            scaled_X_val.append(s.transform(val_i))

        scalers.append(s)
        drop_cols.append(drop_col)
        
    return scaled_X_train, scaled_X_val, scalers, drop_cols

def intensity_rescaling(train):
    df = pd.DataFrame(train, columns = ['pixel_{0}'.format(i+1) for i in range(train.shape[1])])
    
    df[df < 128] = 0
    df[df >= 128] = 1
    
    return df.values

def rescaling(train, val):
    scaled_X_train, scaled_X_val = [], []
    drop_cols = []
    
    pbar = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                       maxval=100).start()
    
    for i in pbar(range(len(train))):
        train_i, drop_col = drop_const_pixel(train[i])
        if val is not None:
            val_i, _ = drop_const_pixel(val[i], drop_col)
        
        scaled_X_train.append(intensity_rescaling(train_i))
        if val is not None:
            scaled_X_val.append(intensity_rescaling(val_i))
        
        drop_cols.append(drop_col)
        
    return scaled_X_train, scaled_X_val, drop_cols

rescaled_models_path = 'rescaled_models.pkl'
assert os.path.isfile(rescaled_models_path)
rescaled_model_list = pickle.load(open(rescaled_models_path, 'rb'))
    
norm_models_path = 'norm_models.pkl'
assert os.path.isfile(norm_models_path)
norm_model_list = pickle.load(open(norm_models_path, 'rb'))

class MNIST_Ensemble_Linear_Log_Reg:
    def __init__(self, ensemblers, weight, norm_scaler, norm_drop_col, rescaled_drop_col):
        self.ensemblers = ensemblers
        assert len(ensemblers) == len(weight)
        self.weight = weight
        self.norm_scaler = norm_scaler
        self.norm_drop_col = norm_drop_col
        self.rescaled_drop_col = rescaled_drop_col
        
    def info(self):
        for i in range(len(self.ensemblers)):
            print(self.ensemblers[i][0], self.ensemblers[i][1], self.weight[i])
        print(self.norm_scaler)
        #print(self.norm_drop_col)
        #print(self.rescaled_drop_col)
        
    def predict(self, X):
        _s_test, _ = drop_const_pixel(X, self.rescaled_drop_col)
        rescaled_x = intensity_rescaling(_s_test)
        
        _n_test, _ = drop_const_pixel(X, self.norm_drop_col)
        norm_x = self.norm_scaler.transform(_n_test)
        
        predictions = []
        for model, _type in self.ensemblers:
            if _type == 0:
                predictions.append(model.predict(norm_x))
            elif _type == 1:
                predictions.append(model.predict(rescaled_x))
            else:
                raise ValueError()
        
        y = []
        for i in range(len(predictions[0])):
            _d = dict.fromkeys(range(10),0)
            
            for j in range(len(self.ensemblers)):
                _d[predictions[j][i]] += self.weight[j]
                
            y.append(max(_d, key=_d.get))
        
        return np.array(y).reshape(len(y),)
    
    def score(self, X, Y):
        y = self.predict(X)
        
        return np.mean(y == Y)

ensemble_model = pickle.load(open('ensemble_model.pkl', 'rb'))

import pygame as pg
import cv2
import tkinter as tk

def pad_image(img, pad_t, pad_r, pad_b, pad_l):
    """Add padding of zeroes to an image.
    Add padding to an array image.
    :param img:
    :param pad_t:
    :param pad_r:
    :param pad_b:
    :param pad_l:
    """
    height, width = img.shape

    # Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    # Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    # Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    # Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img

def center_image(img):
    """Return a centered image.
    :param img:
    """
    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    cropped_image = img[y1:y2, x1:x2]

    zero_axis_fill = (img.shape[0] - cropped_image.shape[0])
    one_axis_fill = (img.shape[1] - cropped_image.shape[1])

    top = int(zero_axis_fill / 2)
    bottom = int(zero_axis_fill - top)
    left = int(one_axis_fill / 2)
    right = int(one_axis_fill - left)

    padded_image = pad_image(cropped_image, top, left, bottom, right)

    return padded_image

img_list = []
pred = []

pg.init()
screen = pg.display.set_mode([512, 512])
pg.display.set_caption("Draw a Number")

radius = 15
black = (0, 0, 0)
isGoing = True
screen.fill((255, 255, 255))
last_pos = (0, 0)

def roundline(srf, color, start, end, radius=2):
    dx = end[0]-start[0]
    dy = end[1]-start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int( start[0]+float(i)/distance*dx)
        y = int( start[1]+float(i)/distance*dy)
        pg.draw.circle(srf, color, (x, y), radius)

pg.font.init()
myFont = pg.font.SysFont("Sans Serif", 10)

draw_on = False
while isGoing:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            isGoing = False

        if event.type == pg.MOUSEBUTTONDOWN:
            spot = event.pos
            pg.draw.circle(screen, black, spot, radius)
            draw_on = True

        if event.type == pg.MOUSEBUTTONUP:
            draw_on = False

        if event.type == pg.MOUSEMOTION:
            if draw_on:
                pg.draw.circle(screen, black, event.pos, radius)
                roundline(screen, black, event.pos, last_pos, radius)
            last_pos = event.pos

        if event.type==pg.KEYDOWN:
            if event.key==pg.K_RETURN:
                pg.image.save(screen, "screenshot.png")
                img = cv2.imread('screenshot.png')

                resized = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
                grayscale = np.dot(resized[...,:3], [0.2989, 0.5870, 0.1140]).reshape(28*28,)
                grayscale = 255 - np.round(grayscale).astype(int) # X
                grayscale = center_image(grayscale.reshape(28, 28)).reshape(28*28,)

                pred_label = ensemble_model.predict(np.array([grayscale]))
                
                img_list.append(grayscale)
                pred.append(pred_label[0])
                
                mess = tk.Tk()
                mess.geometry("350x100+800+450")
                mess.title('What\'s that number?')
                mess.attributes("-topmost", True)
                text = "Is it the number: {0}?".format(pred_label)
                
                l = tk.Label(mess, text=text)
                l.config(font=("Time new Roman", 24))
                l.pack()
                
                def _stop():
                    global isGoing
                    isGoing = False
                    mess.destroy()
                
                # Bug: Nút stop crash ipynb nếu chạy lại cell này :|
                s_mess = tk.Button(mess, text="Stop :(", command=_stop, width=10, height=4)
                s_mess.config(font=("Time new Roman", 14))
                s_mess.pack(padx=5, pady=5, side=tk.LEFT)
                
                c_mess = tk.Button(mess, text="Continue :>", command=mess.destroy, width=10, height=4)
                c_mess.config(font=("Time new Roman", 14))
                c_mess.pack(padx=5, pady=5, side=tk.RIGHT)
                
                mess.mainloop()
                
                screen.fill((255, 255, 255))

        pg.display.flip()

pg.quit()

for i in range(len(img_list)):
    plt.imshow(img_list[i].reshape(28,28), cmap='Greys')
    plt.xlabel(pred[i])
    plt.show()
