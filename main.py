from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Load your dataset
data = pd.read_csv(r"C:\ML-SALARY PREDICTION-project\raw.data.csv", header=None, names=[
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'salary'
])

# Split the dataset into features (X) and target (y)
X = data.drop('salary', axis=1)  # Features
y = data['salary']  # Target variable (income)

# Define the categorical and numerical features
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Preprocessing for numerical features: standard scaling
numerical_transformer = StandardScaler()

# Preprocessing for categorical features: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Column transformer to apply preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Combine preprocessing and model into a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the entire pipeline (model + preprocessing) to a pickle file
with open("salary_prediction_pipeline.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Model training complete and saved to 'salary_prediction_pipeline.pkl'")
