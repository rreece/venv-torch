
VENV_NAME := ".venv"

all: $(VENV_NAME)
	@echo "\nTo start, please run\nsource setup.sh"

$(VENV_NAME):
	bash setup.sh

clean:
	find . -type f -name '*.py[co]' -exec rm -fv {} +
	find . -type d -name __pycache__  -exec rm -rfv {} +
	find . -type d -name .pytest_cache -exec rm -rfv {} +

realclean: clean
	find . -maxdepth 1 -type d -name $(VENV_NAME) -exec rm -rfv {} +
