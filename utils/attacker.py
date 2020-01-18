import random as rd
import string
import pandas as pd
import numpy as np


def changeRandomCharacter(att,permittedPositions):
    random_char = rd.choice(string.ascii_letters+string.digits+string.punctuation)
    new_char_idx = rd.choice(permittedPositions)
    res = att[:new_char_idx]+random_char+att[new_char_idx+1:]
    return res


def attackDataset(dataset,ignore_columns):
    for col in list(dataset):
        if col not in ignore_columns:
            dataset[col] = dataset[col].apply(lambda s: changeRandomCharacter(str(s),
                                                                             np.arange(len(str(s)))))
    return dataset