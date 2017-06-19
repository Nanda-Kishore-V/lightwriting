all:
	#bash run.sh -f "Calligraffiti" -t 'k'
	#bash run.sh -f "Ubuntu" -t 'k'
	bash run.sh -f "Ubuntu" -t 'USC'

clean: clean_src clean_data

clean_src:
	rm -f src/*.pyc
	rm -f src/__pycache__/*

clean_data:
	rm -f data/*.csv
	rm -f data/*.json
	rm -f data/images/*
	rm -f data/segments/*
