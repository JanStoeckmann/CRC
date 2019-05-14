generate and train a new model:
>>python run_unet.py train myNewModel.pkl

train an existing model, the model has to be saved in the model folder:
>>python run_unet.py train myOldModel.pkl

validate an existing model, the model has to be saved in the model folder:
>>python run_unet.py validate myGoodModel.pkl

on Windows its "python" on linux its "python3"