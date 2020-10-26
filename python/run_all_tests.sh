#!/bin/bash
python3 state_space_estimation.py 'rbc' -t 'srivastava' -s 1
python3 state_space_estimation.py 'rbc' -t 'srivastava' -M 3 -n 100 -c 1000 -r 123 -s 1
python3 state_space_estimation.py 'nk' -t 'srivastava' -s 1
python3 state_space_estimation.py 'nk' -t 'srivastava' -M 4 -n 100 -c 1000 -r 123 -s 1
python3 state_space_estimation.py 'real' -t 'srivastava' -s 1
python3 state_space_estimation.py 'rbc' -t 'score' -s 1
python3 state_space_estimation.py 'rbc' -t 'score' -n 100 -c 1000 -r 123 -s 1
python3 state_space_estimation.py 'nk' -t 'score' -s 1
python3 state_space_estimation.py 'nk' -t 'score' -n 100 -c 1000 -r 123 -s 1
python3 state_space_estimation.py 'real' -t 'score' -s 1
python3 state_space_estimation.py 'rbc' -t 'multiple' -s 1
python3 state_space_estimation.py 'rbc' -t 'multiple' -M 3 -n 100 -c 1000 -r 123 -s 1
python3 state_space_estimation.py 'nk' -t 'multiple' -s 1
python3 state_space_estimation.py 'nk' -t 'multiple' -M 4 -n 100 -c 1000 -r 123 -s 1
python3 state_space_estimation.py 'real' -t 'multiple' -s 1
