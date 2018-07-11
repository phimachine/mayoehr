# Post processing 
All post processing script has been added to preprocessing.

## What changed?
* Pandas does not allow NA in integer columns, so we will need to fill NA with 1.

* Split hosp by disch and admit

* surg non collapsed px codes have been deleted for automation purposes. The file is newsurg.csv

# Basic modelling
## Label mystery
How do we evaluate this model? How do we design the loss function?
### Plan A
The model outputs three values for the whole sequence.

* whether there is registered death in our dataset (approxiamtely whether the person is dead for medical reasons)
* why is he dead?
* when is he dead?

The loss is evaluated as:
* registered or not is an independent binary classification
* loss on death is a function of registration. if the death is not registered, the model should receive no backprop loss.
* death date is also a function of registration. if death is not registered, the model should receive no backprop loss. this means all deaths will be in the past. we do not predict natural deaths.

This loss should not bring extra problem. If we do not have loss data, there is not prediction we can learn, so binary classification task befits our data.

The output death cause and time label have no meaning if registration prediction is close to 0. It will be trash value. For any value not close to 0, it can be interpreted as what the machine thinks "the cause of death if the patient is registered in the record".

### Plan B
The model outputs two values for a single timestep.

* is he dead at this time point?
* the cause of death

The advantange of this model is that its output is very homogenous with the input. There is no date necessary. This might be an easier value to process for the model (residual performance).

The disadvantage of this model is that its output is less diverse, and we can do very little to exploit the parallel knowledge. To interpret anything meaningful, we will need to slide the pointer across future timesteps. We can interpret the output as cause of death without caring about "registration flag". Also, the model might be inclined to produce 0 for all outputs, because the dataset is imbalanced. This is not a problem for Plan A.

## my DNC
For experiment purposes, I will use my own DNC implementation and feed the input in.

In the end, it's advantageous to use ixaxaar/pytorch-dnc because I don't want to debug my model.
