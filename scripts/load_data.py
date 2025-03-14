import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocessing(path='data/diamonds.csv'):
    data = pd.read_csv(path)
    
    #Compte des lignes dupliquées et non dupliquées
    data.duplicated().value_counts()
    
    #Suppresion de la column unnamed
    data.drop('Unnamed: 0', axis=1, inplace=True)
    
    #Convertion de type
    data['table'] = data['table'].astype('int')
    
    #encodage des colums
    fe = data.groupby('color').size()/len(data)
    data.loc[:,'Color']=data.color.map(fe)
    data.drop('color',axis=1,inplace=True)
    
    fe = data.groupby('clarity').size()/len(data)
    data.loc[:,'Clarity']=data.clarity.map(fe)
    data.drop('clarity',axis=1,inplace=True)
    
    #Encodage de la column categorielle
    lencoder = LabelEncoder()
    data.cut =lencoder.fit_transform(data.cut)
    
    return data


def del_outliers(data):
    num_col = data.select_dtypes(include=['number'])
    
    for i in num_col:

        Q3 = np.percentile(data[i],75)
        Q1 = np.percentile(data[i],25)
        IQR = Q3 - Q1
        Maxi = Q3 + 1.5*IQR
        Mini = Q1 - 1.5*IQR

        #Ramenons les outliers a la distriution 

        data[i][data[i] > Maxi] = Maxi
        data[i][data[i] < Mini] = Mini
    return data
        
def split_dataset(data):
    #Separation X et Y et transformation en numpy array

    X = data.drop(['cut'],axis=1).values
    y = data['cut'].values
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.20, random_state = 44)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    
    return X_train, X_test, y_train, y_test

def pipeline(path='../data/diamonds.csv'):
    data = preprocessing(path)
    del_outliers(data)
    X_train, X_test, y_train, y_test = split_dataset(data)
    
    return X_train, X_test, y_train, y_test
    