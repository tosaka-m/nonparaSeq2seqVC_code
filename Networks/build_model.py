from .model import S2SVC

def build_model(config={}):
    model = S2SVC(config)
    #model.initialize()
    return model
