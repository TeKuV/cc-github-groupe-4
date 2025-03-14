from trainning_model import train, save_model
from DiamondModel import DiamondModel
from load_data import preprocessing_pipeline
import time


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocessing_pipeline()

    model = DiamondModel(X_train)
    
    model = train(X_train, y_train, X_test, y_test, model)
    save_model(model, f'../models/model_{time.time()}.pth')
    print('Model saved')