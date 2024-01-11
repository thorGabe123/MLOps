import pytest
import random
import sys
from project_name.train_model import parameter, train, dataloader, sample, valid, batch_train
from project_name.models.model import Model

def test_train_epoch():
    model=Model(lr=parameter["learning_rate"], eps=parameter["epsilon"])
    train_dataloader, valid_dataloader = dataloader(model.tokenizer, parameter["batch_size"])
    total_steps = len(train_dataloader) * parameter["epochs"]

    for epoch_i in range(0, parameter["epochs"]):
        # Train model
        train()

        # Get validation loss
        avg_val_loss, validation_time = valid(model,valid_dataloader)
        assert avg_val_loss <= 0

# def test_sample():
#     model
#     sample(model)

#     generated_text = model.generate(bos_token_id=random.randint(1,30000), num_return_sequences=3)
#     assert generated_text is not None

# def test_valid_loss():
#     model, valid_dataloader
#     # Get validation loss
#     avg_val_loss, validation_time = valid(model,valid_dataloader)
#     assert avg_val_loss <= 0

# def test_batch_train_loss():
#     model, batch
#     loss = batch_train(model, batch)
#     assert loss is not None