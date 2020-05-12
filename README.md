# crystal-recognition-ML
Implementation of the crystalization recognition system for plasma crystal based on 'Machine-learning approach for local classification of crystalline structures in multiphase systems'.

'build_crystal_lattice.py' simulates crystal lattice of different structures (fcc, bcc or hcp) with desired level of noise. 
'pre_process.py' calculates the features (from 'features_extraction.py') and distinguishes the boundary particles whose voronoi cell is not complete. 
'filter_and_label.py' filters the disordered particles and does the labelling.
'Ding_model.py' trains the NN (a pause in training is supported by 'Ding_model_continue.py').
