
all:
	@echo Run source setup.sh

clean:
	find . -maxdepth 1 -type d -name venv -exec rm -rfv {} +
	find . -type f -name '*.py[co]' -exec rm -fv {} +
	find . -type d -name __pycache__  -exec rm -rfv {} +
	find . -type d -name .pytest_cache -exec rm -rfv {} +
