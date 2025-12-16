import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X
        self._y = y
        self.train = None
        self.val = None
        self.test = None
        self.label2num = {}
        self.num2label = {}
        self._transform()
    
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''Функция очистки сообщения и преобразования меток в числа.'''
        # Начало вашего кода
        
        
        self.label2num = {"ham": 0, "spam": 1}  
        self.num2label = {0: "ham", 1: "spam"}
        
        new_y = []
        for label in self._y:
            if label == "ham":
                new_y.append(0)
            else:  # spam
                new_y.append(1)
        self._y = new_y
        
        
        clean_x = []
        for text in self._x:
            text = text.lower()  
            text = re.sub(r'[^a-z]', ' ', text)  
            text = ' '.join(text.split())  
            clean_x.append(text)
        
        self._x = clean_x
        
        # Конец вашего кода
    
    def split_dataset(self, val=0.1, test=0.1):
        '''Функция, которая разбивает набор данных на наборы train-validation-test.'''
        # Начало вашего кода
        
        n = len(self._x)
        
        n_test = int(n * test)
        n_val = int(n * val)
        n_train = n - n_test - n_val
        
        
        X_train = self._x[:n_train]
        y_train = self._y[:n_train]
        
        X_val = self._x[n_train:n_train + n_val]
        y_val = self._y[n_train:n_train + n_val]
        
        X_test = self._x[n_train + n_val:]
        y_test = self._y[n_train + n_val:]
        
        self.train = (X_train, y_train)
        self.val = (X_val, y_val)
        self.test = (X_test, y_test)
        
        # Конец вашего кода
        return self.train, self.val, self.test