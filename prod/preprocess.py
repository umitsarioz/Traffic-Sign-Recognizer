import os
import warnings
from pickle import dump, load

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 200)


class Preprocessor:
    def __init__(self, root_filepath: str, train_filename: str, test_filename: str):
        print("Traffic Sign Recognizer Preprocessor Initialized...")
        self.root_fp = root_filepath
        self.train_fp = train_filename
        self.test_fp = test_filename
        self.df_train = self.read_train_data()
        self.df_test = self.read_test_data()

    def read_train_data(self) -> pd.DataFrame:
        filepath = os.path.join(self.root_fp, self.train_fp)
        df = pd.read_csv(filepath)
        return df

    def read_test_data(self) -> pd.DataFrame:
        filepath = os.path.join(self.root_fp, self.test_fp)
        df = pd.read_csv(filepath)
        return df

    def to_categorical(self, val, n_label) -> list:
        arr = np.zeros(n_label)
        arr[val] = 1
        return arr

    def __create_labels(self, df: pd.DataFrame) -> np.ndarray:
        labels = df["ClassId"].sort_values().unique().tolist()
        labels_arr = np.asarray([self.to_categorical(val=val, n_label=len(labels)) for val in df["ClassId"].values])
        return labels_arr

    def create_train_labels(self) -> np.ndarray:
        print("Creating train labels...")
        labels = self.__create_labels(df=self.df_train)
        return labels

    def create_test_labels(self) -> np.ndarray:
        print("Creating test labels...")
        labels = self.__create_labels(df=self.df_test)
        return labels

    def to_pixel(self, image_path, height=32, width=32):
        dim = (height, width)
        img = Image.open(image_path)  # load
        img = img.resize(dim)  # resize
        img = np.asarray(img) / 255.  # normalization
        return img

    def __create_features(self, df: pd.DataFrame) -> np.ndarray:
        filepaths = [os.path.join(self.root_fp, img_path) for img_path in df["Path"].values]
        pixels = np.asarray([self.to_pixel(img_path) for img_path in filepaths])
        return pixels

    def create_train_features(self) -> np.ndarray:
        print("Creating train features...")
        features = self.__create_features(df=self.df_train)
        return features

    def create_test_features(self) -> np.ndarray:
        print("Creating test features...")
        features = self.__create_features(df=self.df_test)
        return features

    def write_pkl(self, data: np.ndarray, filename: str):
        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        print(f"Writing {filename}..")
        with open(filename, 'wb') as file:
            dump(data, file)

    def read_pkl(self, filename: str) -> np.ndarray:
        filename = filename if filename.endswith('.pkl') else filename + '.pkl'
        print(f"Reading {filename}..")
        with open(filename, 'rb') as file:
            arr = load(file)
        return arr

    def run(self, skip_preprocess=True):
        if skip_preprocess:
            print("Skipped preprocess pipeline.")
            return
        else:
            train_features, train_labels = self.create_train_features(), self.create_train_labels()
            test_features, test_labels = self.create_test_features(), self.create_test_labels()
            filenames = ["train_features", "train_labels", "test_features", "test_labels"]
            data = [train_features, train_labels, test_features, test_labels]
            for filename, content in zip(filenames, data):
                filepath = os.path.join(self.root_fp, filename)
                self.write_pkl(data=content, filename=filepath)

    def read_feature_and_labels_from_files(self) -> dict:
        try:
            filenames = ["train_features", "train_labels", "test_features", "test_labels"]
            filepaths = [os.path.join(self.root_fp, filename) for filename in filenames]
            dct = {filename: self.read_pkl(filename=filepath) for filename, filepath in zip(filenames, filepaths)}
            return dct
        except:
            raise Exception("Features.pkl is not found. Run preprocessor using runner.py with skip_preprocess=False parameter.")
