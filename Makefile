SRC_DIR=src
SCRIPT_DIR=shell_scripts
DATA_DIR=data

TEXT=act
FONT=Calligraffiti

all:
	$(SCRIPT_DIR)/run.sh -f "$(FONT)" -t "$(TEXT)"

curve_fitting:
	$(SCRIPT_DIR)/curve_fitting.sh -f "$(FONT)" -t "$(TEXT)"

generate_image:
	$(SCRIPT_DIR)/generate_image.sh -f "$(FONT)" -t "$(TEXT)"

clean: clean_src clean_data

clean_src:
	rm -f $(SRC_DIR)/*.pyc
	rm -r -f $(SRC_DIR)/__pycache__/

clean_data:
	rm -f $(DATA_DIR)/*.csv
	rm -f $(DATA_DIR)/*.json
	rm -f $(DATA_DIR)/images/*
