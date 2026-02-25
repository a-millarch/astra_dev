## This defines all targets as phony targets, i.e. targets that are always out of date
## This is done to ensure that the commands are always executed, even if a file with the same name exists
## See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
## Remove this if you want to use this Makefile for real targets
.PHONY: *

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = astra
PYTHON_VERSION = 3.10.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .
	
# Instal local project as package
local:
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["dev"]

## Delete all compiled Python files
clean:
	$(PYTHON_INTERPRETER) -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	$(PYTHON_INTERPRETER) -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data into processed data
data:
	python $(PROJECT_NAME)/make_data.py


pretrain:
	python $(PROJECT_NAME)/training/train.py --pretrain 
 
train:
	python $(PROJECT_NAME)/training/train.py --pretrain --finetune --eval --multicurve --comprehensive-eval

finetune:
	python $(PROJECT_NAME)/training/train.py --finetune 
	
eval:
	python $(PROJECT_NAME)/training/train.py --eval  --multicurve --comprehensive-eval

sweep_arch:
	python $(PROJECT_NAME)/training/train.py --sweep-arch --n-arch-trials 30
sweep_train:
	python $(PROJECT_NAME)/training/train.py --sweep-train --n-train-trials 50 --eval
sweep:
	python $(PROJECT_NAME)/training/train.py --sweep-arch --sweep-train --pretrain --finetune --eval --comprehensive-eval

## v2: Finetune with transfer learning (pretrain + 4-phase finetune + eval)
train_v2:
	python -m $(PROJECT_NAME).training.train --pretrain --finetune --eval --comprehensive-eval

## v2: Finetune only (no pretrain, uses existing checkpoint)
finetune_v2:
	python -m $(PROJECT_NAME).training.train --finetune --eval --comprehensive-eval

## v2: Two-stage HP sweep (architecture + training)
sweep:
	python -m $(PROJECT_NAME).training.train --sweep-arch --sweep-train --eval --comprehensive-eval

## v2: Architecture sweep only
sweep_arch:
	python -m $(PROJECT_NAME).training.train --sweep-arch --n-arch-trials 30

## v2: Training HP sweep only (requires pretrained checkpoint)
sweep_train:
	python -m $(PROJECT_NAME).training.train --sweep-train --n-train-trials 50 --eval

### Convenience
stash cfg:
	git stash push -m "cfg conflict" configs/defaults.yaml
    
#################################################################################
# Documentation RULES                                                           #
#################################################################################

## File structure print for readme.md
tree:
	make clean
	tree /f

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#   * save line in hold space
#   * purge line
#   * Loop:
#       * append newline + line to hold space
#       * go to next line
#       * if line starts with doc comment, strip comment character off and loop
#   * remove target prerequisites
#   * append hold space (+ newline) to line
#   * replace newline plus comments by `---`
#   * print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')