Deep Learning solution for gender recognition on Fei Dataset 

Several convolutional network models have been built and trained on Fei Dataset. The training and test set are divided in the same way as in paper ["FACE, GENDER AND RACE CLASSIFICATION USING MULTI-REGULARIZED FEATURES LEARNING"](https://pdfs.semanticscholar.org/7cf9/b8a47b078621a4297d0bd3ffde8196d51c8e.pdf) (first 9 images of each person for training, last 5 images for testing) in order to compare VGG-like CNN approach with the approaches in the paper. The paper claims that it achieves 94% accuracy.

Results showed that different VGG-like architectures achieved similar accuracy. With different regularization and initialization settings, the performances of all models range from 92% to 97%.
