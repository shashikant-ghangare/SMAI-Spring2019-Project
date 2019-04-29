# SMAI Spring2019 Project  

## Implement MULTICLASS SVM WITH DIFFERENT KERNELS FROM SCRATCH   
- Team No.      : 44   
- Project ID    : 26
## Team Members  : 

  1. 2018801010 - Karnati Venkata Kartheek    
  2. 2018900061 - Shashikant Ghangare

## Install Dependencies

- Install virtualenv, virtualenvwrapper   
  `sudo pip3 install virtualenv virtualenvwrapper`  
  
- Create a SMCSVM Virtual environment and activate it   
  `mkvirtualenv SMCSVM`   
  `workon SMCSVM`  

- Install required Python Packages   
  `pip3 install -r requiremnts.txt`
 
 ## Usage   
 - Open MCSVM.ipynb for SMCSVM algorithm.
 - To create SMCSVM object use:   
   `clf = SMCSVM()`
 
 - Pass training data and trainng labels to fit() func'tion to train the classifier.   
   `clf.fit(train_X_data, train_y_label)`   
 
 - You can also pass folowing parameters:
    
    1. C, default_value, C=10 - Penalizing factor for Slack  
    2. kernel, default value, kernel='rbf', can also take - 'linear', 'polynomial'  for utilizing kernel tricks on Non-Linear data.   
    3. sigma, default_value=1.0, required for 'RBF' kernel.
    4. degree, default_value=1, degree of polynomial function used in 'polynomial' kernel.
 - To predict on testing data use:    
   `clf.predict(test_X_data)`
 - The algorithm uses K-fold cross validation as a performance metric.
 
 
 ## Run tests
 
 - To run tests, run the run_tests.ipynb file in tests directory
 
 ## References
 - [Support Vector Machines for Multi-Class Pattern Recognition](https://www.researchgate.net/publication/221166057_Support_Vector_Machines_for_Multi-Class_Pattern_Recognition)   
 
 - [Scikit-Learn SVM Class Documentation](https://scikit-learn.org/stable/modules/svm.html)   
 
