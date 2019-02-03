What's the plan?
Well, because the intersection of the drug users and drug related deaths is so low, the correlation cannot be learnt.
Only 1% of the dataset contains signals. Bad.

1, Go in REP dataset, redo prep by converting all codes to ICD10
* This is a very significanet work. It is going to take at least 20 hours.

2, Redo drug user and deaths selection with R or pandas. Not InputGen.

3, Modify the InputGen for new data columns and dictionary generations. Back test the drug codes.

4, So the goal of the task is to predict drug related deaths. We know what drug this person takes,
and we want to predict the chance of him dying.
* Again, missing death records is going to be a significant problem. That is going to suppress the prediction signals.
* Predicting which drug this drug user will die of does not seem clinically useful. Of course it's the drug
that the person is taking? The chance is the goal. However, seeing if the network can see the connections is interesting.
Can we query the network to associate which drug to which drug? An algorithm can be developed. Get a sample, get the
gradient of each input. Average the gradient over multiple samples. We can see the association of input to outputs.
Especially in this case where we know there is a relationship, this strategy should work.