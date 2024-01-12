import pytest
import random
import sys
from project_name.train_model import parameter, train, dataloader, sample, valid, batch_train
from project_name.models.model import Model
from hydra import initialize, compose

def test_valid():
    with initialize(version_base=None, config_path="../project_name"):
        cfg = compose(config_name="config")
        model = Model(lr=cfg["hyperparameters"]["learning_rate"], eps=cfg["hyperparameters"]["epsilon"])
        tokenizer = model.tokenizer
        model.resize_token_embeddings(len(tokenizer))
        train_dataloader, valid_dataloader = dataloader(tokenizer, parameter["batch_size"])
        # Get validation loss
        avg_val_loss, validation_time = valid(model,valid_dataloader)
        assert avg_val_loss <= 0
        assert validation_time > 0