import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split






train_path = "boston-housing/train.csv"
test_path = "boston-housing/test.csv"

# loading the data 
train_df = pd.read_csv(train_path)

# features and target

# for features we don't need the ID and Target column
x_train_full = train_df.drop(columns=['ID','medv'])

# we just need the target column
y_train_full = train_df['medv']

# spliting the data 
# we did like 20% for testing and 80% for training
x_train_split,x_test_split,y_train_split,y_test_split = train_test_split(x_train_full,y_train_full,test_size=0.2,random_state=42)

# dividing the features
numerical_features = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'black', 'lstat']
categorical_features=['chas', 'rad']

# Transform 
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore',drop='first')

# creating the preprocessor using columntransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numerical_features),
        ('cat',categorical_transformer,categorical_features)
    ],
    remainder='passthrough'
)


# processing the data 
x_train_processed = preprocessor.fit_transform(x_train_split)
x_test_processed = preprocessor.transform(x_test_split)


# training the model using SGDRegresor

sgd_reg = SGDRegressor(
    max_iter=1000,
    eta0=0.01,
    random_state=42
)

sgd_reg.fit(x_train_processed,y_train_split)


# testing the model

y_train_pred = sgd_reg.predict(x_train_processed)
y_test_pred = sgd_reg.predict(x_test_processed)


mse_train = mean_squared_error(y_train_split,y_train_pred)
mse_test = mean_squared_error(y_test_split,y_test_pred)

print(f"MSE (TRAIN) (SGD) : {mse_train}")
print(f"MSE (TEST) (SGD): {mse_test}")



import joblib

model_filename = 'model.joblib'
preprocessor_filename = 'preprocessor.joblib'

joblib.dump(sgd_reg, model_filename)
joblib.dump(preprocessor, preprocessor_filename)

print("✅ Model and preprocessor saved successfully!")