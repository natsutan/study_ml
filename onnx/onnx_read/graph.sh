#!/bin/sh
python ~/gitproj/outside/onnx/onnx/tools/net_drawer.py --input squeezenet.onnx --output squeezenet.dot --embed_docstring
dot  -Tpdf squeezenet.dot -o squeezenet.pdf
