#!/bin/bash

for dir in */; do
    echo "$dir"
    cd $dir
    ffmpeg -framerate 15 -i frame_%03d.png -q:v 1 ${dir%/}.gif -y
    cd ..
done

