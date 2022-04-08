PYTHON38=/opt/homebrew/Cellar/python@3.8/3.8.13/bin/python3.8

venv:
	${PYTHON38} -m venv venv

clean:
	-rm -r venv
	-rm -r __pycache__

deps: venv/lib/bitarray

venv/lib/bitarray: venv requirements.txt
	venv/bin/pip3 install -r requirements.txt

run: venv deps
	venv/bin/python meteor.py
