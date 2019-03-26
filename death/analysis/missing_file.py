

from pathlib import Path

a=Path("adnc.csv")
with a.open('r') as f:
    for line in f:
        print(line)

print(a.absolute())