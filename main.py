import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype
from scipy import stats

dataframe = pd.read_csv(sys.argv[1])
print("Welcome!!", '\n',"Columns are: ", '\n')
for i in dataframe.columns:
    print(i, end=' ')
print("\nType: ", dataframe.columns.dtype)
targets = input("Enter target column: ")
target = dataframe[targets]

print("Target column: \n")
print(target, '\n')

def data_description():
    global dataframe
    print("1. single column\n2. Entire dataset\n3. show columns\n")
    option1 = int(input("Enter choice (-1 for previous): "))

    if option1 == 1:
        col = input("Enter column name: ")
        print(dataframe[col].describe())

    elif option1 == 2:
        print(dataframe.describe(), '\n')
        print(dataframe.info())

    elif option1 == 3:
        num = int(input("Number of rows: "))
        print(dataframe.iloc[:num, :])

    elif option1 == -1:
        main()

    data_description()

def null_values():
    global dataframe
    print("1. Total null values\n2. Remove columns\n3. Fill Null with Mean\n4. Fill Null with Median\n5. Fill Null with Mode\n6. Show dataset")
    nulls = ['', 'Nan', 'NULL', 'N/A', 'NA', '?']
    dataframe = dataframe.replace(nulls, np.nan)
    option1 = int(input("Enter choice (-1 for previous): "))

    if option1 == 1:
        print(dataframe.isnull().sum())

    elif option1 == 2:
        name = input("enter coulumn name: ")
        name = list(name.split(' '))
        dataframe = dataframe.drop(name, axis=1)
        print(dataframe)

    elif option1 == 3:
        for i in dataframe.columns:
            if dataframe[i].isnull().sum() > 0:
                dataframe[i].replace(np.nan, dataframe[i].mean(), inplace=True)

    elif option1 == 4:
        for i in dataframe.columns:
            if dataframe[i].isnull().sum() > 0:
                dataframe[i].replace(np.nan, dataframe[i].median(), inplace=True)
                
    elif option1 == 5:
        for i in dataframe.columns:
            if dataframe[i].isnull().sum() > 0:
                dataframe[i].replace(np.nan, dataframe[i].mode(), inplace=True)

    elif option1 == 6:
        print(dataframe)

    elif option1 == -1:
        main()

    null_values()

def encoding_data():
    global dataframe
    global target
    global targets
    print("1. Show all categorical data\n2. Perform hot encoding\n3. Show dataset")
    option1 = int(input("Enter the choice (-1 for previous): "))

    if option1 == 1:
        cat_col = dataframe.select_dtypes(include=['category', 'object'])
        print(cat_col.head())

    elif option1 == 2:
        if target.dtype == 'object':
            target = pd.DataFrame(LabelEncoder().fit_transform(target))
            print(target)
        x = pd.DataFrame(dataframe.drop(targets, axis=1))
        if len(x.select_dtypes(include=['category', 'object']).columns) > 0:
            cat_f = x.select_dtypes(include=['category', 'object'])
            num_f = x.select_dtypes('number')
            cat_f = pd.get_dummies(cat_f)
            x = pd.concat([cat_f, num_f], join='inner', axis=1)
        print(x)
        dataframe.loc[:, dataframe.columns != targets] = x
        dataframe[targets] = target
        print(dataframe)

    elif option1 == 3:
        print(dataframe)

    elif option1 == -1:
        main()

    encoding_data()

def normalization():
    global dataframe
    print("1. Specific column\n2. Whole dataset\n3. Show dataset")
    option2 = int(input("Enter choice (-1 for previous): "))

    if option2 == 1:
        col = input("Enter column name: ")

        if col in dataframe.columns:
            dataframe[col] = dataframe[col] / dataframe[col].abs().max()
            print(dataframe[col])

    elif option2 == 2:
        for i in dataframe.columns:
            if is_numeric_dtype(dataframe[i]):
                dataframe[i]=(dataframe[i]-dataframe[i].min())/(dataframe[i].max()-dataframe[i].min())
        print(dataframe)

    elif option2 == 3:
        print(dataframe)

    elif option2 == -1:
        feature_scaling()

    normalization()

def standardization():
    global dataframe
    print("1. Specific column\n2. Whole dataset\n3. Show dataset")
    option2 = int(input("Enter choice (-1 for previous): "))

    if option2 == 1:
        col = input("Enter column name: ")

        if col in dataframe.columns:
            dataframe[col] = stats.zscore(dataframe[col])
            print(dataframe[col])

    elif option2 == 2:
        for i in dataframe.columns:
            if is_numeric_dtype(dataframe[i]):
                dataframe[i] = stats.zscore(dataframe[i])
        print(dataframe)

    elif option2 == 3:
        print(dataframe)

    elif option2 == -1:
        feature_scaling()

    standardization()

def feature_scaling():
    global dataframe
    print("1. Perform Normalization(MinMax Scaler)\n2. Perform Standardization\n3. Show dataset")
    option1 = int(input("Enter the choice (-1 for previous): "))

    if option1 == 1:
        normalization()

    elif option1 == 2:
        standardization()

    elif option1 == 3:
        print(dataframe)

    elif option1 == -1:
        main()

    feature_scaling()

def main():
    global dataframe
    print("Note: You can exit here only elsewhere inside u cannot exit\n")
    print("1. Data description\n2. Handling Null values\n3. Encoding Categorical data\n4. Feature scacling of data\n5. Download the preprocessed dataset\n6.Exit\n")
    option = int(input("Please choose the preprocessing task: "))
    if option == 1:
        data_description()

    elif option == 2:
        null_values()

    elif option == 3:
        encoding_data()

    elif option == 4:
        feature_scaling()

    elif option == 5:
        name = input("Enter the name of the file: ")
        dataframe.to_csv(f"{name}.csv", index=False)

    elif option == 6:
        exit(0)

    main()

main()