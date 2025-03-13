import torch.nn as nn
from load_data import preprocessing, del_outliers, split_dataset
from standardisation import to_tensor, standardisation

class DiamondModel(nn.Module):
    
    X_train, X_test, y_train, y_test = split_dataset(del_outliers(preprocessing()))
    
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=to_tensor(standardisation()).shape[1], out_features=15)
        self.layer_2 = nn.Linear(in_features=15, out_features=12)
        self.layer_3 = nn.Linear(in_features=12, out_features=8)
        self.layer_4 = nn.Linear(in_features=8, out_features=5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x