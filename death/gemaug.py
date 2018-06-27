# noticing that the gem diagnosis conversion code has many rows missing, we will need to augment the data

import os
import pathlib
from pathlib import Path

fpath=Path('/infodev1/home/m193194/git/ehr/death/data/2018_I10gem.txt')
with open(fpath,"r") as f:
    line=f.readline()
    print(line)
