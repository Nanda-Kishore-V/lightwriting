all:
	shell_scripts/run.sh -f "Calligraffiti" -t 'act'

generate_image:
	shell_scripts/generate_image.sh -f "Ubuntu" -t 'text'

clean: clean_src clean_data

clean_src:
	rm -f src/*.pyc
	rm -r -f src/__pycache__/

clean_data:
	rm -f data/*.csv
	rm -f data/*.json
	rm -f data/images/*
