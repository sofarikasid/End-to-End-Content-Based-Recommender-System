install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest test_run_rec_sys.py

format:
	black *.py
	black mylib/*.py

lint:
	# run lint checks on all python files in the directory and subdirectories
	
	#pylint --disable=R,C *.py
	#pylint --disable=W,C --ignored-modules=module_name mylib/*.py
	pylint --disable=R,C --ignored-modules=module_name *.py
	

all: install test format