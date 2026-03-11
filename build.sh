#!/bin/bash
set -e &&           \
rm -rf ./dist &&    \
python -m build &&  \
pip install ./dist/*.whl --force-reinstall 