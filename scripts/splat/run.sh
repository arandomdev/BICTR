#!/usr/bin/env bash
set -e

if [ $# -eq 0 ]; then
    echo "Please provide site name, ex, run.sh siteA"
    exit 1
fi

site=$1

splat-hd \
    -t "./transmitters/${site}.qth" \
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
rm -f "${site}.zip"
7zz a "${site}.zip" map.kml map.ppm map-ck.ppm splat.dcf

# prepare kmz for google earth
sed -i -e 's|<href>map.ppm</href>|<href>map.png</href>|g' map.kml
sed -i -e 's|<href>map-ck.ppm</href>|<href>map-ck.png</href>|g' map.kml
sed -i -e 's|<overlayXY x="0" y="1" xunits="fraction" yunits="fraction"/>|<overlayXY x="1" y="0.5" xunits="fraction" yunits="fraction"/>|g' map.kml
sed -i -e 's|<screenXY x="0" y="1" xunits="fraction" yunits="fraction"/>|<screenXY x="1" y="0.5" xunits="fraction" yunits="fraction"/>|g' map.kml

rm -f "${site}_google.zip"
7zz a "${site}_google.zip" map.kml map.png map-ck.png
mv "${site}_google.zip" "${site}.kmz"