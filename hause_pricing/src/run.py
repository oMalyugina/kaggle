from Preprocessor import Reader
from Model import Model

config = {"reader": {
    "scaler": "robust_scaler",
    "encoder": "label_encoder"
},
    "model": {
        "model": "random_forest",
        "params": ""
    }
}

reader = Reader(config["reader"])
X, y = reader.read_data("your path")
train_model = Model(config["model"])
test_model = Model(config["model"])

# your training and testing
