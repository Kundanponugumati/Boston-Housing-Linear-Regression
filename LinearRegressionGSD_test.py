import joblib
import pandas as pd

model = joblib.load('model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

test_df = pd.read_csv("boston-housing/test.csv")

x_test = test_df.drop(columns=['ID'])
x_processed = preprocessor.transform(x_test)

y_pred = model.predict(x_processed)
print(y_pred)