all:
	bash run.sh -f "Ubuntu" -t 'Happy'

clean: clean_src clean_data

clean_src:
	rm -f src/*.pyc
	rm -r -f src/__pycache__/

clean_data:
	rm -f data/*.csv
	rm -f data/*.json
	rm -f data/images/*
	rm -f data/segments/*

limit_width:
	bash run.sh -t "aovawjvea;jckajvlekajvdklsjdcjfdkcafnvdvdcsdajdkjsadvasc" -f 'Ubuntu'

limit_height:
	bash run.sh -t "aov\nawj\nvea;j\nckajv\nl\nekajvdk\nlsjdc\njfdk\nca\nfnv\ndvdcs\ndaj\ndkjs\nadvasc" -f 'Ubuntu'
