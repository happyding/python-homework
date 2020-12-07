#### Resampling
Use the above to answer the following:

> Which model had the best balanced accuracy score?
  Naive Random Oversampling: 0.832780705572896
  SMOTE Oversampling:        0.8388510243681058
  Cluster Centroids Undersampling: 0.8215575767118339
  Combination (Over and Under) Sampling: 0.8388319216626994

  SMOTE Oversampling has the best balanced accuracy score.

> Which model had the best recall score?
  They all have recall scores between 0.81 to 0.88. They are all very close.

> Which model had the best geometric mean score?
  Naive Random Oversampling: 0.83
  SMOTE Oversampling:        0.84
  Cluster Centroids Undersampling: 0.82
  Combination (Over and Under) Sampling: 0.84

  They are all very close. SMOTE Oversampling and Combination (Over and Under) Sampling have the best balanced accuracy scores.
  


#### Ensemble Learning
Use the above to answer the following:

> Which model had the best balanced accuracy score?
  Easy Ensemble Classifier has the better balanced accuracy score (0.931601605553446) than Balanced Random Forest Classifier (0.7855345052746622)
>
> Which model had the best recall score?
  Easy Ensemble Classifier has the better recall score than Balanced Random Forest Classifier.
>
> Which model had the best geometric mean score?
  Easy Ensemble Classifier has the better geometric mean score than Balanced Random Forest Classifier.
>
> What are the top three features?
  total_rec_prncp : ( 0.09175752102205247 )
  total_pymnt_inv : ( 0.06410003199501778 )
  total_pymnt : ( 0.05764917485461809 )
