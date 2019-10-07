from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import pandas as pd


class Preprocessor:
    def __init__(self, params: dict):
        self.params = params
        if params["scaler"] == "standert_scaler":
            self.scaler = StandardScaler()
        elif params["scaler"] == "robust_scaler":
            self.scaler = RobustScaler()
        else:
            print("wrong scaler parameters")
            raise KeyError

        self.encoder = {}
        if params["encoder"] == "label_encoder":
            self.base_encoder = LabelEncoder
        else:
            print("wrong encoder parameters")
            raise KeyError

    def fit_transform(self, X_old):
        X = X_old.copy()
        for var_name in X.select_dtypes(include=['object']):
            encoder = self.base_encoder
            encoder.transform(X[var_name].astype(str))
            X[var_name] = encoder.fit(X[var_name].astype(str))
            self.encoder[var_name] = encoder
        self.scaler.fit(X)
        return self.scaler.transform(X)

    def fit(self, X_old):
        X = X_old.copy()
        for var_name in X.select_dtypes(include=['object']):
            X[var_name] = self.encoder[var_name].fit(X[var_name].astype(str))
        return self.scaler.transform(X)


class Reader:

    def __init__(self, config):
        self.preprocessor = Preprocessor(config)

    def read_data(self, csv_file_name: str, y_name:str):
        data = pd.read_csv(csv_file_name)
        self.y = data[y_name]
        self.X = data.loc[:, data.columns != 'y_name']
        self.X = self.preprocessor.fit_transform(self.X)

        return self.X, self.y

