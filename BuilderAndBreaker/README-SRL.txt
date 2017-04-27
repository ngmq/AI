=== DATA SOURCE ===

This task and data are derived from the He+Lewis+Zettlemoyer's work on Question-Answer Driven Semantic Role Labeling (https://dada.cs.washington.edu/qasrl/). 

The task is, given a sentence and a question related to one of the predicates in the sentence, output the span of the sentence that answers the question.


=== DATA FORMAT ===

The training data includes one file in TSV (tab separated values) format containing the following:
#id: A unique identifier
prefix: Part of the sentence before the predicate
predicate: The predicate
suffix: Part of the sentence after the predicate
question: Question
answer: Answer

Note: Although the training data is the same as the one for the QASRL task, the file format is slightly different from the original file format. 
