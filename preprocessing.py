import pandas as pd
import subprocess
import sys
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
def combine_csv(file1, file2):
    '''
    Combines two csv files with same column info into one dataframe
    '''
    data_frame1 = []
    data_frame2 = []
    if os.path.exists(file1):
        print( file1 + " found ")
        data_frame1 = pd.read_csv(file1, index_col=0)
    else:
        print( "file not found.")
        return

    if os.path.exists(file2):
        print( file2 + " found ")
        data_frame2 = pd.read_csv(file2, index_col=0)
    else:
        print("file not found.")
        return

    full_data = pd.concat([data_frame1, data_frame2], ignore_index=True)
    return full_data
def get_csv_data(filename, index=0):
    '''
    Gets data from a csv file and puts it into a pandas dataframe
    '''
    if os.path.exists(filename):
        print( filename + " found ")
        data_frame = pd.read_csv(filename, index_col=index)
        return data_frame
    else:
        print("file not found")
def create_inputs(dataframe, target, num_features=31):
    '''
    Splits dataframe into features and target values
    dataframe - full datatable
    target - name of the column that is the target
    '''
    features = list(dataframe.columns[:num_features])
    #print(features)
    y = dataframe[target]
    X = dataframe[features]
    scaler = StandardScaler()
    scaler.fit(X.values)
    X = scaler.transform(X.values)
    return X, y.values
def encode_features(features, dataframe, reduce_classes=True, binary_classes=False):
    '''
    encodes features that are string values to integers
    features- features to encode
    dataframe - full pandas data table
    reduce_classes-reduces classes down to 4 (student dataset)
    '''
    dataframe_copy = dataframe.copy()
    le = LabelEncoder()
    for feature in features:
        dataframe_copy[feature] = le.fit_transform(dataframe_copy[feature])

    if reduce_classes:
        for i in range(0, len(dataframe_copy["G3"].values)):
            if dataframe_copy["G3"].values[i] >= 0 and dataframe_copy["G3"].values[i] <=4:
                dataframe_copy["G3"].values[i] = 0
            elif dataframe_copy["G3"].values[i] >= 5 and dataframe_copy["G3"].values[i] <=10:
                dataframe_copy["G3"].values[i] = 1
            elif dataframe_copy["G3"].values[i] >= 11 and dataframe_copy["G3"].values[i] <= 15:
                dataframe_copy["G3"].values[i] = 2
            elif dataframe_copy["G3"].values[i] >= 16 and dataframe_copy["G3"].values[i] <= 20:
                dataframe_copy["G3"].values[i] = 3
    if binary_classes:
        for i in range(0, len(dataframe_copy["G3"].values)):
            if dataframe_copy["G3"].values[i] >=0 and dataframe_copy["G3"].values[i] <= 10:
                dataframe_copy["G3"].values[i] = 0
            else:
                dataframe_copy["G3"].values[i] = 1
    return dataframe_copy
def wine_binary_classes(dataframe):
    for i in range(0, len(dataframe)):
        if dataframe["quality"].values[i] >=0  and dataframe["quality"].values[i] <=5:
            dataframe["quality"].values[i] = 0
        elif dataframe["quality"].values[i] >= 6 and dataframe["quality"].values[i] <= 10:
            dataframe["quality"].values[i] = 1
    return dataframe
