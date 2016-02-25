# coding=utf-8
import codecs
from os import listdir
import os
from os.path import isdir, join

import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.utils import check_random_state

__author__ = 'rockychi1001@gmail.com'


def load_files(container_path, description=None, categories=None,
               shuffle=True, encoding='utf-8', random_state=0,
               key_path_index=-2):
    """Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        container_folder/
            category_1_folder/
                file_1.txt
                    line 1
                    line 2
                    ...
                    line n
            category_2_folder/
                file_2.txt
                    line 1
                    line 2
                    ...
                    line n
            ...

    The folder names are used as supervised signal label names. The
    individual file names are not important.

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category

    description: string or unicode, optional (default=None)
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    categories : A collection of strings or None, optional (default=None)
        If None (default), load all the categories.
        If not None, list of category names to load (other categories ignored).

    shuffle : bool, optional (default=True)
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    key_name_index : int, category's index containing text file.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: either
        data, the raw text data to learn, or 'filenames', the files
        holding it, 'target', the classification labels (integer index),
        'target_names', the meaning of the labels, and 'DESCR', the full
        description of the dataset.
    """
    target = list()
    target_names = list()
    filenames = list()
    filelines2data = dict()

    folders = [f for f in sorted(listdir(container_path))
               if isdir(join(container_path, f))]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        documents = [join(folder_path, d)
                     for d in sorted(listdir(folder_path))]
        for training_doc in documents:
            if key_path_index:
                category = training_doc.split(os.sep)[key_path_index]
            else:
                category = training_doc
            with codecs.open(training_doc, encoding=encoding) as td:
                for line_index, data in enumerate(td):
                    key4file = category + str(line_index)
                    filelines2data[key4file] = data
                    target.append(label)
                    filenames.append(key4file)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    data = list()
    for filename in filenames:
        data.append(filelines2data.get(filename))

    return Bunch(data=data,
                 filenames=filenames,
                 target_names=target_names,
                 target=target,
                 DESCR=description)
