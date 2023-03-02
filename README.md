# ThalamusDB: Answering Complex Queries with Natural Language Predicates on Multi-Modal Data

## Quick Start

```bash
# Git LFS required to download *.zip
unzip craigslist/furniture_imgs.zip -d craigslist
# Python 3
pip install -r requirements.txt
python benchmark.py
```

Note that, to run the YouTube benchmark, you would need to download and setup the audio model: https://github.com/akoepke/audio-retrieval-benchmark. Then, uncomment the relevant code in `nldbs.py`. If you just want to run the Craigslist benchmark, replace the YouTube benchmark queries in `benchmark.py` with an empty list.

## How to Integrate New Datasets

Create a new function in `nldbs.py` that creates a `NLDatabase` instance (refer to functions `craisglist()` and `youtubeaudios()`). It requires loading relational data to DuckDB and providing pointers to image and audio data.

## Demo Video

https://youtu.be/wV9UhULhFg8
