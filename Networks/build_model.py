from .model import S2SVC

def build_model(model_params={}):
    model = S2SVC(model_params)
    return model
