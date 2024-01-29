import os.path

import numpy as np
import tensorflow.keras as K
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, lr: float | int, n_epoch: int, width: int, height: int, n_label: int, model_name='ai_model.h5'):
        self.lr = lr
        self.n_epoch = n_epoch
        self.width = width
        self.height = height
        self.n_label = n_label
        self.model_name = os.path.join('models', model_name)
        self.model = None
        self.history = None
        print("Trainer class is initialized...")
        self.__check_or_create_models_folder()

    def __check_or_create_models_folder(self):
        model_path = os.path.join('models')
        if not os.path.exists(model_path):
            os.mkdir(model_path)

    def __create_model(self):
        print("Creating model...")
        model = K.models.Sequential([
            K.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                            input_shape=(self.height, self.width, 3)),
            K.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            K.layers.MaxPool2D(pool_size=(2, 2)),
            K.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            K.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            K.layers.MaxPool2D(pool_size=(2, 2)),
            K.layers.Flatten(),
            K.layers.Dense(512, activation='relu'),
            K.layers.Dropout(rate=0.5),
            K.layers.Dense(self.n_label, activation='softmax')
        ])

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, save=False):
        model = self.__create_model()
        opt = K.optimizers.legacy.Adam(lr=self.lr, decay=self.lr / (self.n_epoch * 0.5))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        print("Training is starting...")
        history = model.fit(X_train, y_train, epochs=self.n_epoch, validation_data=(X_val, y_val))

        self.model = model
        self.history = history
        if save:
            print(f"Saving model as name {self.model_name}")
            self.model.save(self.model_name)

    def load_model(self):
        try:
            print(f"Loading prediction model as name {self.model_name}")
            return K.models.load_model(self.model_name)
        except:
            raise Exception("Model file is not found. Train model using runner.py with skip_train=False.")

    def plot_acc_loss(self, save=False):
        if self.history:
            plt.plot(self.history.history['loss'], label='Train_Loss')
            plt.plot(self.history.history['val_loss'], label='Val_Loss')
            plt.plot(self.history.history['accuracy'], label='Train_Acc')
            plt.plot(self.history.history['val_accuracy'], label='Val_Acc')
            plt.legend(loc='best')
            plt.show()
            if save:
                filepath = os.path.join('images', 'model_scores.png')
                plt.savefig(filepath, bbox_inches='tight')
        else:
            raise Exception("No history object is exist. First you need to train a model to plot.")

    def calculate_accuracy_score(self, model, x, y) -> float:
        y_pred = np.argmax(model.predict(x), axis=1)
        y_true = np.argmax(y, axis=1)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        return acc * 100
