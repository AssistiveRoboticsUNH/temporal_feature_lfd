Preparation:

1. Define source files in XX.txt
2. Train the spatial feature learner: ./train_spatial.sh
3. Generate temporal features: ./prepare_ditrl.sh
	- ./generate_raw_iad.sh
	- ./threshold_iad.sh
	- ./convert_iad_to_itr.sh
4. Train temporal feature learner: ./train_ditirl.sh
5. Evaluate temporal feature learner: ./evaluate_ditrl.sh