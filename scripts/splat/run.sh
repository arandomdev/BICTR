#!/usr/bin/env bash
set -e

splat-hd \
    -t ./transmitters/SplatTx.qth \
    -L 2 \
    -d ./sdf \
    -R 2.5 \
    -metric \
    -m 1.333 \
    -dbm \
    -kml \
    -ngs \
    -o map.ppm

# Convert to images
convert -transparent "#FFFFFF" map.ppm map.png
convert map-ck.ppm map-ck.png

# First prepare zip with ppm files
rm -f splat_lwchm.zip
7zz a splat_lwchm.zip map.kml map.ppm map-ck.ppm splat.dcf

# prepare kmz for google earth
sed -i -e 's|<href>map.ppm</href>|<href>map.png</href>|g' map.kml
sed -i -e 's|<href>map-ck.ppm</href>|<href>map-ck.png</href>|g' map.kml
sed -i -e 's|<overlayXY x="0" y="1" xunits="fraction" yunits="fraction"/>|<overlayXY x="1" y="0.5" xunits="fraction" yunits="fraction"/>|g' map.kml
sed -i -e 's|<screenXY x="0" y="1" xunits="fraction" yunits="fraction"/>|<screenXY x="1" y="0.5" xunits="fraction" yunits="fraction"/>|g' map.kml

rm -f splat_google.zip
7zz a splat_google.zip map.kml map.png map-ck.png
mv splat_google.zip splat_google.kmz