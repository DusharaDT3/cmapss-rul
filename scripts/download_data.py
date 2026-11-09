"""Download NASA C-MAPSS turbofan data into data/raw/.

Source: NASA Prognostics Center of Excellence data repository (PHM Society mirror).
Citation: Saxena, A., Goebel, K., Simon, D. and Eklund, N. (2008) 'Damage propagation
modeling for aircraft engine run-to-failure simulation', PHM08, Denver, CO.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = [f"{p}_FD00{i}.txt" for p in ("train", "test", "RUL") for i in range(1, 5)]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / f).exists() for f in EXPECTED):
        print("Data already present in", RAW_DIR)
        return
    print("Downloading", URL)
    with urllib.request.urlopen(URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for name in inner.namelist():
        if name.endswith(".txt"):
            (RAW_DIR / Path(name).name).write_bytes(inner.read(name))
    missing = [f for f in EXPECTED if not (RAW_DIR / f).exists()]
    if missing:
        sys.exit(f"Missing files after extraction: {missing}")
    print("Saved files to", RAW_DIR)


if __name__ == "__main__":
    main()
