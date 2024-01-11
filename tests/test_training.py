import torch
import sys
sys.path.append('project_name/models')
from omegaconf import OmegaConf
import pytest

def get_param():
    config = OmegaConf.load('project_name/config.yaml')
    batch_size = config['hyperparameters']['batch_size']
    epochs = config['hyperparameters']['epochs']
    warmup_steps = config['hyperparameters']['warmup_steps']
    sample_every = config['hyperparameters']['sample_every']
    return batch_size, epochs, warmup_steps, sample_every

def test_init_lr():
    lr,eps = get_param()
    model = Model(lr=lr)
    assert model.lr == lr

def test_init_epsilon():
    lr,eps = get_param()
    model = Model(eps=eps)
    assert model.epsilon == eps

# def test_forward_pass():
#     model = Model()
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     assert outputs is not None

# def test_forward_pass_invalid_inputs(input_ids, attention_mask):
#     model = Model()
#     with pytest.raises(ValueError):
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# def test_generate_text(input_text):
#     model = Model()
#     generated_text = model.generate(input_text)
#     assert isinstance(generated_text, str)

# def test_generate_text_empty_input(input_text):
#     model = Model()
#     with pytest.raises(ValueError):
#         generated_text = model.generate(input_text)

# def test_configure_optimizers(lr, epsilon):
#     model = Model(lr=lr, epsilon=epsilon)
#     optimizer = model.configure_optimizers()
#     assert isinstance(optimizer, torch.optim.AdamW)

