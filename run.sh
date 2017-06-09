#!/bin/bash

# converts text to trajectories

# t = text to be shown
# f = font to be used
while getopts f:t: option
do
	case "${option}"
	in
	t) text=${OPTARG};;
	f) font=${OPTARG};;
	esac
done

pointsize=500

# create image with required text
convert -background white -font $font -size 2000x2000 -fill black -pointsize $pointsize -gravity center label:$text -trim -bordercolor "#FFF" -border 10 data/images/temporary.bmp

python src/image_preprocessing.py

echo 'image generated from text!'

python src/image_to_waypoints.py
echo 'done with image_to_waypoints.py'

#python3 src/generation.py
python3 src/generation.py
echo 'done with generation.py'
