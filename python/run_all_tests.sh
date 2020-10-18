#!/bin/bash
python3 state_space_estimation.py 'rbc' -t -s 1
python3 state_space_estimation.py 'rbc' -n 100 -c 1000 -s 1
python3 state_space_estimation.py 'nk' -s 1
python3 state_space_estimation.py 'nk' -m 4 -n 100 -c 1000 -s 1
python3 state_space_estimation.py 'real' -t 'custom_4' -s 1