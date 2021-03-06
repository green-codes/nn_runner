import torch

import numpy as np
import os
import pickle
import cv2
import gzip
import random
import csv


class SimpleDataset(torch.utils.data.Dataset):
    """
    Simple, all-purpose, map-style dataset class. Loads data from disk
    on-demand if given file lists, else loads all data at init.

    This class currently supports reading npy, pkl, and csv data files
    directly. Gzip compressed archives are supported. If provided with lists
    of data file paths, this class can also read all image formats supported
    by OpenCV (cv2) at load time.  

    If caching is enabled, the dataset attempts to save all read (and 
    optionally preprocessed) data in system memory, which may cause out of
    memory errors for large datasets. 

    - X: Requires path to either the x data file, or a text file containing a
        list of paths for x data. 
    - Y: Requires path to either the y data file, or a text file containing a
        list of paths for y data. 
    - Data transformers should be callables that return processed data
        individually.

    NOTE: this dataset class does NOT handle batching, which is expected to be
        handled by the pytorch DataLoader class, or other samplers. 

    NOTE: to use bulk reading, pre-make the bulk data files w.r.t. desired
        batches and obtain a path list, then use this class with DataLoader
        batching disabled.

    NOTE: the current implementation cannot handle very large text files, given
        limited system memory. Handling very large text files require
        streaming, which we won't implement as it doesn't make sense with a
        map-style dataset. Make a iterable dataset if we need streaming. 
    """

    def __init__(self, x_file, y_file=None, x_transform=None, y_transform=None,
                 use_cache=False, x_path_prefix=None, y_path_prefix=None):
        """
        Parameters
        - x_file: input data file, or text file w/list of paths
        - y_file: label data file, or text file w/list of paths
        - x_transform: callable preprocessor for x
        - y_transform: callable preprocessor for y
        - use_cache: whether to cache loaded data in memory
        - x_path_prefix: path prefix to x data files
        - y_path_prefix: path prefix to y data files
        """

        self.x_transform = x_transform
        self.y_transform = y_transform
        self.use_cache = use_cache
        self.x_path_prefix = x_path_prefix
        self.y_path_prefix = y_path_prefix
        self.cache_dict = {}

        # read x file
        if x_file.endswith(".npy"):
            self.x_arr = self._load_numpy(x_file)
        elif x_file.endswith(".pkl"):
            self.x_arr = self._load_pickle(x_file)
        elif x_file.endswith(".csv"):
            self.x_arr = self._load_csv(x_file)
        else:
            with open(x_file, 'r') as f:
                self.x_arr = [e.strip() for e in f.readlines()]

        # read y file
        if y_file is not None:
            if y_file.endswith(".npy"):
                self.y_arr = self._load_numpy(y_file)
            elif y_file.endswith(".pkl"):
                self.y_arr = self._load_pickle(y_file)
            elif y_file.endswith(".csv"):
                self.y_arr = self._load_csv(y_file)
            else:
                with open(y_file, 'r') as f:
                    self.y_arr = [e.strip() for e in f.readlines()]
            assert len(self.x_arr) == len(self.y_arr)  # sanity check
        else:
            self.y_arr = None

    def __len__(self):
        return len(self.x_arr)

    def __getitem__(self, index):
        if self.use_cache and index in self.cache_dict:
            return self.cache_dict[index]

        x = self.x_arr[index]
        if isinstance(x, str):
            if self.x_path_prefix:  # now assume that x is a path
                x = os.path.join(self.x_path_prefix, x)
            if os.path.exists(x):  # load from file
                if ".npy" in x:
                    x = self._load_numpy(x)
                elif ".pkl" in x:  # pickle file, assume ndarray
                    x = self._load_pickle(x)
                elif ".csv" in x:
                    x = self._load_csv(x)
                else:              # assume file in image format
                    x = cv2.imread(x)
        if self.x_transform is not None:
            x = self.x_transform(x)

        if self.y_arr is not None:
            y = self.y_arr[index]
            if isinstance(y, str):  # load from file
                if self.y_path_prefix:
                    y = os.path.join(self.y_path_prefix, y)
                if os.path.exists(y):
                    if ".npy" in y:
                        y = self._load_numpy(y)
                    elif ".pkl" in y:  # pickle file, assume ndarray
                        y = self._load_pickle(y)
                    elif ".csv" in y:
                        y = self._load_csv(y)
                    else:            # assume file in image format
                        y = cv2.imread(y)
            if self.y_transform is not None:
                y = self.y_transform(y)
            if self.use_cache:
                self.cache_dict[index] = (x, y)
            return x, y
        else:
            if self.use_cache:
                self.cache_dict[index] = x
            return x

    def _shuffle(self):
        shuffled_idx = np.random.permutation(self.__len__())
        self.x_paths = self.x_paths[shuffled_idx]
        if self.y_arr is not None:
            self.y_arr = self.y_arr[shuffled_idx]

    @staticmethod
    def _unzip(path):
        """ unzip if necessary, returns an open file """
        path = path.strip()
        return gzip.open(path) if path.endswith(".gz") else open(path, 'rb')
        # TODO: add other uncompression methods

    @staticmethod
    def _load_numpy(path):
        f = SimpleDataset._unzip(path)
        return np.load(f)

    @staticmethod
    def _load_pickle(path):
        f = SimpleDataset._unzip(path)
        ret = pickle.load(f)
        assert isinstance(ret, np.ndarray), "Expected numpy.ndarray"
        return ret

    @staticmethod
    def _load_csv(path):
        f = SimpleDataset._unzip(path)
        ret = [e for e in csv.reader(f.readlines())]
        f.close()
        return ret
        # NOTE: we cannot infer data types (eg text data) so do not attempt to
        # remove csv header or convert to numpy
