# noticing that the gem diagnosis conversion code has many rows missing, we will need to augment the data

import os
import pathlib
from pathlib import Path

fpath=Path('/infodev1/home/m193194/git/ehr/death/data/2018_I10gem.txt')
iimap={}

with open(fpath,"r") as f:
    for line in f:
        ret=line.split()
        iimap[ret[0]]=ret[1]

iimap

print('done')