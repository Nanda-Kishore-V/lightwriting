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

# create image with required text
convert -background white -gravity center -size 3000x750 -font "$font" label:"$text" -threshold 50% data/images/temporary.bmp
echo 'image generated from text!'

python src/image_to_waypoints.py
echo 'done with image_to_waypoints.py'

python3 src/generation.py
echo 'done with generation.py'

python src/form_pairs.py
echo 'done combining segments into longer segments'

python src/split_into_planes.py
echo 'done splitting segements into different planes.'
