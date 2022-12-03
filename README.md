# venv-torch

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
