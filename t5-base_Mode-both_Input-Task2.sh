#!/bin/bash
python3 -u  main.py \
	-model t5-base \
	-tt both \
	-epochs 1 \
	-input task \
	--beam 5 \
	-bs 16
