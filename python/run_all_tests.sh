#!/bin/bash
python3 state_space_estimation.py 'rbc' -s 1
python3 state_space_estimation.py 'rbc' -n 100 -c 1000 -s 1
python3 state_space_estimation.py 'nk' -s 1
python3 state_space_estimation.py 'nk' -M 4 -n 100 -c 1000 -s 1
<<<<<<< HEAD
python3 state_space_estimation.py 'real' -s 1
=======
python3 state_space_estimation.py 'real' -s 1
python3 state_space_estimation.py 'real' -n 1000 -s 1
>>>>>>> 0a54cc19aa78d7d5fe5605f0257c78cf368efafe
