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

## Create the conda environment (CPU; use environment_gpu.yml on GPU machines)
create_environment:
	conda env create -f environment_cpu.yml

## Create the conda environment (GPU)
create_environment_gpu:
	conda env create -f environment_gpu.yml

## Install the package into the active environment (deps come from the conda env)
requirements:
	$(PYTHON_INTERPRETER) -m pip install -e .

# Instal local project as package
local:
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install with the REST-service extra (fastapi/uvicorn)
service_requirements:
	$(PYTHON_INTERPRETER) -m pip install -e .[service]

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
	python $(PROJECT_NAME)/training/train.py --eval  --multicurve --comprehensive-eval --active-only

sweep:
	python $(PROJECT_NAME)/training/train.py --sweep --finetune --eval --comprehensive-eval

## v2: Finetune with transfer learning (pretrain + 4-phase finetune + eval)
train_v2:
	python -m $(PROJECT_NAME).training.train --pretrain --finetune --eval --comprehensive-eval

## v2: Finetune only (no pretrain, uses existing checkpoint)
finetune_v2:
	python -m $(PROJECT_NAME).training.train --finetune --eval --comprehensive-eval

## Joint HP sweep (architecture + training) → pretrain best → retrain full trainval
sweep_v2:
	python -m $(PROJECT_NAME).training.train --sweep --finetune --eval --comprehensive-eval

## Train EBM models at all time intervals
ebm_models:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).models.ebm.generate_ebm_feature

## Delete trained EBM models
clear_ebm_models:
	rm -rf models/ebm/*

## Full CPU pipeline: data → EBM models → data cache (overwrites all)
all_cpu:
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/make_data.py --overwrite
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).models.ebm.generate_ebm_feature
	$(PYTHON_INTERPRETER) -c "from astra.utils import cfg, setup_logging; setup_logging(); from astra.data.caching import prepare_data_and_dls_cached; prepare_data_and_dls_cached(cfg, force_refresh=True)"

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
build_documentation:
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation:
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