from project_name.train_model import parameter, train, dataloader, sample, valid, batch_train, format_time
from project_name.models.model import Model
from hydra import initialize, compose
import time
import re

def test_parameter():
    with initialize(version_base=None, config_path="../project_name"):
        cfg = compose(config_name="config")
        assert parameter['epochs'] == cfg["hyperparameters"]["epochs"]
        assert parameter['learning_rate'] == cfg["hyperparameters"]["learning_rate"]
        assert parameter['warmup_steps'] == cfg["hyperparameters"]["warmup_steps"]
        assert parameter['epsilon'] == cfg["hyperparameters"]["epsilon"]
        assert parameter['batch_size'] == cfg["hyperparameters"]["batch_size"]
        assert parameter['sample_every'] == cfg["hyperparameters"]["sample_every"]

def test_format_time():
    elapsed = format_time(time.time())
    assert type(elapsed) == str
    assert re.search(r"\d*:\d*:\d*", elapsed)

def test_training():
    model = Model()
    model.training = 0
    model.train()
    assert model.training == 1