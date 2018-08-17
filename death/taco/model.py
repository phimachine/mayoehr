# RNN reads a whole sequence, and then outputs the target sequences at <GO> frame.
# If you think about it this way, can't we read the whole sequence and output the goal at <GO> frame just once?
# Our goal is to produce a target sequence of length one.
# In this case, would attention be necessary?