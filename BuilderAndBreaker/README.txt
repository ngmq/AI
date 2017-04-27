=== DATA SOURCE ===
This is the dataset of the paper:
Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)

Each sentence in this dataset corresponds to a review from a pool of Rotten Tomatoes reviews. Please read the paper for details on extracting a sentence from a review

=== DATA FORMAT ===

The training data includes two files in TSV (tab separated values) format:
	a. sentences contains sentence IDs, sentences, sentiment labels and positivity probabilities separated by tabs 
	b. phrases contains phrase IDs, phrases, sentiment labels and positivity probabilities separated by tabs
   Please note that phrase ids and sentence ids are not the same. A phrase can be a part of multiple sentences

=== SENTIMENT LABELS ===

Each sentence is given a sentiment label of -1 or +1 corresponding to negative or positive, respectively
Each sentence has been parsed into a parse tree & and each of its phrases has been given a sentiment label (negative or positive)
We provide the phrase level sentiment labels (which you are free to use or ignore)

Each sentence is also associated with a positivity probability which is a number between 0 and 1
The two classes can be mapped to the positivity probability using the following cut-offs:
[0, 0.4] ==> negative
[0.6, 1.0] ==> positive

NOTE: All neutral sentences i.e. sentences with positivity probability between (0.4, 0.6) in the original dataset were removed while creating this dataset since in our task we were interested in only positive and negative sentiments.
