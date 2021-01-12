# RMOTE
RSMOTE is proposed in  *RSMOTE: a self-adaptive robust SMOTE for imbalanced problems with label noise* .

Imbalanced classification is a common task in supervised learning. RSMOTE aims to generate synthetic samples to make the dataset more balanced. 
This method remove the noise from minority samples, and divides the rest minority samples into two parts, the borderline and the safe, based on the 2-means clustering result of relative density of minority samples. Synthetic samples are generated within each part. The number of synthetic samples generated from each minority sample, depends on the relative density of it. The relative densit is defined as follows:
$2+3/5$
