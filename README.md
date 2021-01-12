# RMOTE
RSMOTE is proposed in  *RSMOTE: a self-adaptive robust SMOTE for imbalanced problems with label noise* .

Imbalanced classification is a common task in supervised learning. RSMOTE aims to generate synthetic samples to make the dataset more balanced.   
This method remove the noise from minority samples, and divides the rest minority samples into two parts, the borderline and the safe, based on the 2-means clustering result of relative density of minority samples. Synthetic samples are generated within each part. The number of synthetic samples generated from each minority sample, depends on the relative density of it. That is, more new samples are generate from the safe samples, and less are from the borderline samples. This will make the borderline between majority and minority samples more clear.  

This r package allows users to generate synthetic samples from imbalanced data with two categories.
