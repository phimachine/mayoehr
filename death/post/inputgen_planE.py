"""
1, This is a new input gen plan that puts batch processing on a high priority
The idea is to append sequences one after one within batches. For example, one particular sequence
reaches an end at early stage, then we simply append another new sequence.
This normalization should be performed at input level. There are difficulties implementing weights
BN.

2, This also aims to solve some validation-training split problem should training restart.
Instead of pickling the pandas dataset, I should pickle the training, validation, test.

"""