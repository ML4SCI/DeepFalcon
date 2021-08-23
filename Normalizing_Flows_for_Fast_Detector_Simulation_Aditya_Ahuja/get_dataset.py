#!/bin/bash

mkdir -p data
cd data
wget "https://cernbox.cern.ch/index.php/s/xcBgv3Vw3rmHu9u"
mv download 'Boosted_Jets_Samples.snappy.zip'
unzip Boosted_Jets_Samples.snappy.zip
rm Boosted_Jets_Samples.snappy.zip