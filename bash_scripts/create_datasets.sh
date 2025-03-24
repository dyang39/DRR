#!/bin/bash

python data/prepare_commonsenseqa.py
python data/prepare_openbookqa.py
python data/prepare_piqa.py
python data/prepare_winogrande.py