#!/bin/bash

# DO NOT RUN THIS SCRIPT EXCEPT IN GOOGLE COLAB
# OTHERWISE IT MAY BREAK YOUR ENVIRONMENT

# Get the required R packages (pre-built)
gdown 1FJDSNJc2tGE5etUWS_NwYJefmAwBOz1q
unzip extra_r_packages.zip -d /
rm extra_r_packages.zip

# Download the data
wget -O anon_gps_tracks_with_dive.zip https://www.dropbox.com/s/idm5zzqik7qpmwc/anon_gps_tracks_with_dive.csv.zip?dl=0
unzip anon_gps_tracks_with_dive.zip
rm anon_gps_tracks_with_dive.zip
