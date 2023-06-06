install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest test_run_rec_sys.py

format:
	black *.py
	black mylib/*.py

lint:
	pylint --disable=R,C --extension-pkg-whitelist='pydantic' mylib/main.py --ignore-patterns=test_.*?py *.py mylib/*.py

all: install test format