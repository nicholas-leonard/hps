"""
A Pylearn2 Dataset object for accessing the data for the
Kaggle facial-keypoint-detection contest for the IFT 6266 H13 course.
"""
__authors__ = 'Vincent Archambault-Bouffard'
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "Vincent Archambault-Bouffard"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import os
import csv
import numpy as np

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.string_utils import preprocess

# The number of features in the Y vector
numberOfKeyPoints = 30


class FacialKeypoint(DenseDesignMatrix):
    """
    A Pylearn2 Dataset object for accessing the data for the
    Kaggle facial-keypoint-detection contest for the IFT 6266 H13 course.
    """

    def __init__(self, which_set,
                 start=None,
                 stop=None,
                 axes=('b', 0, 1, 'c')):
        """
        which_set: A string specifying which portion of the dataset
            to load. Valid values are 'train' or 'public_test'
        base_path: The directory containing the .csv files from kaggle.com.
                   If you are using this on the DIRO filesystem, you
                   can just use the default value. If you are using this
                   at home, you should download the .csv files from
                   Kaggle and set base_path to the directory containing
                   them.
        fit_preprocessor: True if the preprocessor is allowed to fit the
                   data.
        fit_test_preprocessor: If we construct a test set based on this
                    dataset, should it be allowed to fit the test set?
        """

        files = {'train': 'keypoints_train.csv', 'public_test': 'keypoints_test.csv'}

        try:
            filename = files[which_set]
        except KeyError:
            raise ValueError("Unrecognized dataset name: " + which_set)

        path = os.path.join("${KEYPOINTS_DATA_PATH}/", filename)
        path = preprocess(path)
        csv_file = open(path, 'r')

        reader = csv.reader(csv_file)

        # Discard header
        row = reader.next()

        y_list = []
        X_list = []

        for row in reader:
            if which_set == 'train':
                y_float = readKeyPoints(row)
                X_row_str = row[numberOfKeyPoints]  # The image is at the last position
                y_list.append(y_float)
            else:
                _, X_row_str = row
            X_row_strs = X_row_str.split(' ')
            X_row = map(lambda x: float(x), X_row_strs)
            X_list.append(X_row)

        X = np.asarray(X_list)
        if which_set == 'train':
            y = np.asarray(y_list)
        else:
            y = None

        if start is not None:
            assert which_set != 'public_test'
            assert isinstance(start, int)
            assert isinstance(stop, int)
            assert start >= 0
            assert start < stop
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            if y is not None:
                y = y[start:stop, :]

        view_converter = DefaultViewConverter(shape=[96, 96, 1], axes=axes)

        super(FacialKeypoint, self).__init__(X=X, y=y, view_converter=view_converter)

    def adjust_for_viewer(self, X):
        return (X - 127.5) / 127.5



def readKeyPoints(row):
    """
    Reads the list of keypoints from a row in the csv file
    """
    kp = [-1] * numberOfKeyPoints
    for i in range(numberOfKeyPoints):
        if row[i] is not None and row[i] != "":
            kp[i] = float(row[i])
    return kp
