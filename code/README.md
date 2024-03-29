# meteor-thesis

Python code for my bachelor's thesis regarding extensions of the Meteor stegosystem

This repository is a fork
of [the original Meteor Colab Notebook](https://colab.research.google.com/gist/tusharjois/ec8603b711ff61e09167d8fef37c9b86).
Meteor is a steganographic system which uses a generative neural network to hide hidden text in a ML-based covert channel.

This is very much research code and MUST NOT be used in production in any way.
If you have questions regarding my modifications to the Meteor stegosystem, feel free to reach out.
You can reach me at mail (at) jboy.eu.

# REQUIREMENTS

- python3 (tested with python3.8 on macOS 12.3.1) with venv module

# INSTALL

1. clone this repository: `git clone https://github.com/seasox/meteor-thesis`
2. a.) If your system has `make` installed, use `make run` to run the `meteor_symmetric.py` file.

   b.) If your system does not have `make` installed, consult the contents of `Makefile`. It is recommended to create a
   new `venv` using `python3 -m venv venv` to keep all the deps in place. Afterwards, install dependencies
   using `venv/bin/pip3 install -r requirements`
   Now, run `venv/bin/python3 meteor_symmetric.py` to run the `meteor_symmetric.py` file.

