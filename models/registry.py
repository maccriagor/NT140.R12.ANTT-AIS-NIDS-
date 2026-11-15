"""
Model registry shared by skeleton.py and pipeline.py.

Every model file in models/ registers itself with @register("<name>"); build_model(name, n_classes) in
skeleton.py then looks the class up in MODELS and instantiates it as MODELS[name](n_classes).
"""
N_BYTES = 1500          # payload bytes per packet = model input width

MODELS = {}


def register(name: str):
    def deco(cls):
        if name in MODELS:
            raise KeyError(f"model name '{name}' is already registered by {MODELS[name].__name__}")
        MODELS[name] = cls
        return cls
    return deco
