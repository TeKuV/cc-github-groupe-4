from scripts.DiamondModel import DiamondModel
from scripts.load_data import preprocessing_pipeline
from scripts.trainning_model import train, save_model
from scripts.standardisation import standardisation, to_tensor
import time


X_train, X_test, y_train, y_test = preprocessing_pipeline("./data/diamonds.csv")

X_train = to_tensor(standardisation(X_train))
X_test = to_tensor(standardisation(X_test))
y_train = to_tensor(y_train)
y_test = to_tensor(y_test)

model = DiamondModel(X_train.shape[1])

model = train(X_train, y_train, X_test, y_test, model)
save_model(model, f"./models/model_final.pth")
print('Model saved')