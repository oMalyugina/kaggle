from sklearn.ensemble import RandomForestRegressor


class Model:
    def __init__(self, config):
        if config["model"] == "random_forest":
            self.model = RandomForestRegressor(**config["params"])
        else:
            print("wrong model parameters")
            raise KeyError

    def my_fit(self, X, y):
        pass

    def my_test(self, X):
        pass
