# Group 30 - Text Generation using Natural Language Processing

## Overall goal of the project
The goal of the project is to fine-tune a model to generate comprehensive paragraphs continuing on an initial text prompt.

## What framework are you going to use?
We will be working with NLP, so we plan to use the Transformers framework. This selected based on Transformer having packages that contain pretrainted model with tokenizer, that can be used in our project. We Will also be using a dataset on huggingface, so we will be using the Transformer framework to handle data processing, which should make it for a better process that handling it using pandas and torch alone.

## How do you intend to include the framework into your project
We expect to start with a pretrained model and fine-tune it by training it on our data.

## What data are you going to run on?
We utilize a dataset of publicly available text books as found from https://huggingface.co/datasets/izumi-lab/open-text-books containing raw texts of the books. The books in this dataset differ in content and should be good for creating a model that will generate text that would be suitable for new books.

## What deep learning models do you expect to use?
We are going to use a pretrained GPT2 text-generating model as found at https://huggingface.co/gpt2, and fine-tune it by training on our dataset. The Transformer framework will be used in this process since the model is taking for their framework, we will also be using the optimzer from the Transformer framework.


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── project_name  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
