install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval test_rocchio_classify.py

format:
	black *.py
	black mylib/*.py

lint:
	# run lint checks on all python files in the directory and subdirectories
	pylint --disable=R,C mylib/*.py
	pylint --disable=R,C *.py
	

		


all: install lint test