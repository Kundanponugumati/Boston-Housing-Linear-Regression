# We are predicting the value of medv (target variable)

1. EDA
2. Feature engineering 
3. Training 
4. Testing


first we read the data_set and make it as train_df

from that we divide into 2 parts 
1. features (x_full)
2. variable (y_full)

now we split dataset in 2 parts 
1. training (x_train) (y_train)
2. testing  (x_test) (y_test)

now we transformering the data before passing to model 
x_train -> x_train_processed
x_test -> x_test_processed

now we giving x_train_processed,y_train_processed to model to learn

