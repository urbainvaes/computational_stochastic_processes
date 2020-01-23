PYTHON_SCRIPTS := $(wildcard w*.py)
IPYNB_OUTPUTS := $(PYTHON_SCRIPTS:.py=.ipynb)
HTML_OUTPUTS := $(PYTHON_SCRIPTS:.py=.html)

all : $(HTML_OUTPUTS) $(IPYNB_OUTPUTS)

%.ipynb : %.py
	# jupytext --to 'notebook' $^
	~/.local/bin/jupytext --to 'notebook' $^
	jupyter nbconvert --execute --to 'notebook' --inplace $@

%.html : %.ipynb
	jupyter nbconvert --to html $^

clean :
	rm $(IPYNB_OUTPUTS) $(HTML_OUTPUTS)
