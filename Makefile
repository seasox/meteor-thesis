PYTHON3=python3

venv:
	${PYTHON3} -m venv venv

clean:
	-rm -r venv
	-rm -r __pycache__

deps: venv/lib/bitarray

venv/lib/bitarray: venv requirements.txt
	venv/bin/pip3 install -r requirements.txt

run: venv deps
	venv/bin/python meteor_symmetric.py
