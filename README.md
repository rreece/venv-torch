# venv-torch

[![CI badge](https://github.com/rreece/venv-torch/actions/workflows/ci.yml/badge.svg)](https://github.com/rreece/venv-torch/actions)

An example of setting up a python virtualenv that has pytorch
installed.

This package also shows examples of loading and using HuggingFace
models. So far, all the models used are for NLP.

An analogous project using poetry to manage the virtualenv and its
dependencies is
[poetry-torch](https://github.com/rreece/poetry-torch).


## How to setup

```
source setup.sh
```


## Run sentiment scoring of your input texts

```
cd tests
python test_hf_bert_sentiment.py
```


## Run the tests

```
pytest
```

## Author

Ryan Reece ([@rreece](https://github.com/rreece))         
Created: November 4, 2022
