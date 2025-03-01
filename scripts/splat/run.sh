#!/usr/bin/env bash
set -e

if [ $# -eq 0 ]; then
    echo "Please provide site name, ex, run.sh siteA"
    exit 1
fi

site=$1

# Splat is unable open the lrp files from the transmitter folder
# Need to work around it
cp "./transmitters/${site}.lrp" ./splat.lrp

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

# Prepare output dir
mkdir -p artifacts

# Convert to images
convert -transparent "#FFFFFF" map.ppm map.png
convert map-ck.ppm map-ck.png

# First prepare zip with ppm files
rm -f "./artifacts/${site}.zip"
7zz a "./artifacts/${site}.zip" map.kml map.ppm map-ck.ppm splat.dcf

# prepare kmz for google earth
sed -i -e 's|<href>map.ppm</href>|<href>map.png</href>|g' map.kml
sed -i -e 's|<href>map-ck.ppm</href>|<href>map-ck.png</href>|g' map.kml
sed -i -e 's|<overlayXY x="0" y="1" xunits="fraction" yunits="fraction"/>|<overlayXY x="1" y="0.5" xunits="fraction" yunits="fraction"/>|g' map.kml
sed -i -e 's|<screenXY x="0" y="1" xunits="fraction" yunits="fraction"/>|<screenXY x="1" y="0.5" xunits="fraction" yunits="fraction"/>|g' map.kml

rm -f "./artifacts/${site}_google.zip"
7zz a "./artifacts/${site}_google.zip" map.kml map.png map-ck.png
mv "./artifacts/${site}_google.zip" "./artifacts/${site}.kmz"

# Remove temporary files
rm map-ck.png
rm map-ck.ppm
rm map.kml
rm map.png
rm map.ppm
rm splat.lrp
rm *-site_report.txt