PYTHON_SCRIPTS := $(wildcard w*.py)
IPYNB_OUTPUTS := $(PYTHON_SCRIPTS:.py=.ipynb)
HTML_OUTPUTS := $(PYTHON_SCRIPTS:.py=.html)

all : $(HTML_OUTPUTS) $(IPYNB_OUTPUTS)

%.ipynb : %.py
	# jupytext --to 'notebook' $^
	~/.local/bin/jupytext --to 'notebook' $^
	JUPYTER_CONFIG_DIR=.jupyter jupyter nbconvert --execute --to 'notebook' --inplace $@

%.html : %.ipynb
	JUPYTER_CONFIG_DIR=.jupyter jupyter nbconvert --to html $^

clean :
	rm $(IPYNB_OUTPUTS) $(HTML_OUTPUTS)
