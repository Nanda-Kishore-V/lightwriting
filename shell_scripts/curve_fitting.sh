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
bash shell_scripts/generate_image.sh -t "$text" -f "$font"

python src/image_to_curve_fitting.py
echo 'done with image_to_curve_fitting.py'

python src/form_pairs.py
echo 'done combining segments into longer segments'

python src/split_into_planes.py
echo 'done splitting segements into different planes.'
