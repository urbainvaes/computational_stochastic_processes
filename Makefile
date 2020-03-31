PYTHON_SCRIPTS := $(wildcard w*.py) \
    coursework_problem_1.py \
    coursework_problem_2.py \
    coursework_problem_3.py
IPYNB_OUTPUTS := $(PYTHON_SCRIPTS:.py=.ipynb)
HTML_OUTPUTS := $(PYTHON_SCRIPTS:.py=.html)
CSS := .jupyter/custom/custom.css

all : $(HTML_OUTPUTS) $(IPYNB_OUTPUTS)

%.ipynb : %.py $(CSS)
	# jupytext --to 'notebook' $^
	~/.local/bin/jupytext --to 'notebook' $<
	JUPYTER_CONFIG_DIR=.jupyter jupyter nbconvert --execute --to 'notebook' --inplace $@

%.html : %.ipynb $(CSS)
	JUPYTER_CONFIG_DIR=.jupyter jupyter nbconvert --to html $<

clean :
	rm $(IPYNB_OUTPUTS) $(HTML_OUTPUTS)
