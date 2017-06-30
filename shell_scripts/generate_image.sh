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
convert -background white -gravity center -size 3000x750 -font "$font" label:"$text" -threshold 50% data/images/black_text.bmp
echo 'image generated from text!'
