SRC_DIR=src
SCRIPT_DIR=shell_scripts
DATA_DIR=data

TEXT=USC
FONT=Ubuntu

.PHONY: all
all:
	$(SCRIPT_DIR)/run.sh -f "$(FONT)" -t "$(TEXT)"

.PHONY: curve_fitting
curve_fitting:
	$(SCRIPT_DIR)/curve_fitting.sh -f "$(FONT)" -t "$(TEXT)"

.PHONY: generate_image
generate_image:
	$(SCRIPT_DIR)/generate_image.sh -f "$(FONT)" -t "$(TEXT)"

.PHONY: clean
clean: clean_src clean_data

.PHONY: clean_src
clean_src:
	rm -f $(SRC_DIR)/*.pyc
	rm -r -f $(SRC_DIR)/__pycache__/

.PHONY: clean_data
clean_data:
	rm -f $(DATA_DIR)/*.csv
	rm -f $(DATA_DIR)/*.json
	rm -f $(DATA_DIR)/images/*
