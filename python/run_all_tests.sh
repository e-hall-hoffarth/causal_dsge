#!/bin/bash
python3 state_space_estimation.py 'rbc' -s 1
python3 state_space_estimation.py 'rbc' -n 100 -c 1000 -s 1
python3 state_space_estimation.py 'nk' -s 1
python3 state_space_estimation.py 'nk' -M 4 -n 100 -c 1000 -s 1
python3 state_space_estimation.py 'real' -t -s 1
