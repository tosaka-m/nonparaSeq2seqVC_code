from .model import VCS2S

def build_model(model_params={}):
    model = VCS2S(model_params)
    return model
