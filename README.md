# Student Performance

Author: Ben Gruher
Date: 12/2/2021

## Problem description

This problem focuses on predicting a student's performance in school based on a number of characteristics related to the student's lifestyle, personal traits, and current living situation. The dataset focuses on student performance in two subjects, mathematics and Portuguese. Schools may be interested in a model that can accurately predict a student's performance so that they can direct resources towards students who are less likely to be successful in school in order to prevent those students from dropping out, failing classes, or falling behind their peers. 

The original paper describing this problem can be found [here](http://www3.dsi.uminho.pt/pcortez/student.pdf). 

Our project will differ from the original paper in three ways. First, we aim to predict the final grade with high accuracy without needing the first and second period (or semester) grades. The authors of the original paper found that knowledge of the first and/or second period grades greatly improved the accuracy of the models. However, in a lot of cases, this information may be discovered at a point where it is too late for the school to help the student pass the class. As a result, we will remove the student's previous grades from consideration when building our model. Second, we want our solution to be able to generalize for schools outside of the two Portuguese schools from which the data was collected. Therefore, we will be removing the school code from considerations as well. Third, we want our solution to be able to generalize for all subjects, not just math and Portuguese, so we will be combining the two datasets into a single dataset.

The original paper also describes several approaches to solving the problem including regression, binary classification, and multi-class classification. We will be focusing on regression and hope to improve upon the results of the best regression model in the paper (3.90 RMSE for math and 2.67 RMSE for Portuguese). 

We have also included a repository containing the steps to run the project in the cloud using Amazon Sagemaker. This process involves creating an Inference Pipeline containing two models, one for the preprocessing steps and one for the inference. These models are fit to the training data and deployed to a managed endpoint. The endpoint accepts data that has not been preprocessed and returns a prediction. The models support both csv and JSON formats. See the aws/ folder for more details.

## Dataset

This dataset is courtesy of the UC Irvine Machine Learning Repository and was donated in 2014. The dataset contains non-identifiable information on students from two Portuguese secondary schools. You can find the dataset and read more [here](https://archive.ics.uci.edu/ml/datasets/Student+Performance). 

The columns of the dataset are defined as follows:

Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:
1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
2. sex - student's sex (binary: 'F' - female or 'M' - male)
3. age - student's age (numeric: from 15 to 22)
4. address - student's home address type (binary: 'U' - urban or 'R' - rural)
5. famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
6. Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
7. Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
8. Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
9. Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10. Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
11. reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12. guardian - student's guardian (nominal: 'mother', 'father' or 'other')
13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15. failures - number of past class failures (numeric: n if 1<=n<3, else 4)
16. schoolsup - extra educational support (binary: yes or no)
17. famsup - family educational support (binary: yes or no)
18. paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19. activities - extra-curricular activities (binary: yes or no)
20. nursery - attended nursery school (binary: yes or no)
21. higher - wants to take higher education (binary: yes or no)
22. internet - Internet access at home (binary: yes or no)
23. romantic - with a romantic relationship (binary: yes or no)
24. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25. freetime - free time after school (numeric: from 1 - very low to 5 - very high)
26. goout - going out with friends (numeric: from 1 - very low to 5 - very high)
27. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29. health - current health status (numeric: from 1 - very bad to 5 - very good)
30. absences - number of school absences (numeric: from 0 to 93)

These grades are related with the course subject, Math or Portuguese:
31. G1 - first period grade (numeric: from 0 to 20)
32. G2 - second period grade (numeric: from 0 to 20)
33. G3 - final grade (numeric: from 0 to 20, output target)


You can also download the data into a folder called 'data/' by running the cells in the DataIngestion.ipynb notebook (see below for instructions).

## Running the notebooks

To execute the cells in the notebook, press Shift-Enter when a cell is highlighted. You can also select Cells->Run All from the drop down menu to run the entire notebook. 

Note that for the Amazon Sagemaker example, the notebook will need certain AWS IAM permissions and will need the ability to create a Sagemaker session. The easiest way to do this is to run the notebook on an Amazon Sagemaker managed notebook instance.
