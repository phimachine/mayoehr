# final test loads all models and run the test dataset

# Reviewed InputGen. I think this design works:
# For cn models, the model reads the whole sequence, and only the prediction at the last timestep will be evaluated
# For sequence models, the model reads the whole sequence, and the prediction loss will be averaged
# Besides loss, other metrics should be collected.