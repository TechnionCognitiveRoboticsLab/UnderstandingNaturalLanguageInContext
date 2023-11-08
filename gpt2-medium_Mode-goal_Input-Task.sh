#!/bin/bash
python3 -u  main.py \
	-model gpt2-medium \
	-tt goal \
	-epochs 1 \
	-input task \
	--beam 5 \
	-bs 16
