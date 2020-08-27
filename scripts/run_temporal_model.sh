MODEL_FLAGS = --ditrl --trim
MODELNAME = 

python3 train_model.py bs $(MODEL_FLAGS) --modelname $(MODELNAME) | 
python3 eval_model.py bs $(MODEL_FLAGS) --modelname $(MODELNAME)
