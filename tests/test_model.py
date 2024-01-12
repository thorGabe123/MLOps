import torch
from project_name.models.model import Model
from hydra import initialize, compose

def test_initialize():
    with initialize(version_base=None, config_path="../project_name"):
        cfg = compose(config_name="config")
        model = Model()
        assert cfg['hyperparameters']['epsilon'] == model.epsilon
        assert cfg['hyperparameters']['learning_rate'] == model.lr 