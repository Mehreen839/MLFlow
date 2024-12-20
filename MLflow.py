# Create Your Initial Project in MLflow
import mlflow 

# Set the a tracking URI to a local sqlite file
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")

# In MLflow create a new Experiment 
experiment_id = mlflow.create_experiment("PotentialStartups")
# Print the Experiment Name and Creation Date
experiment = mlflow.get_experiment(experiment_id)
print("Name: {}".format(experiment.name))
print("Creation timestamp: {}".format(experiment.creation_time))
# Run an MLFlow UI for a Visual 
!mlflow server --backend-store-uri="sqlite:///mydb.sqlite"
# Begin with loading the Dataset into Training Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the Dataset
df = pd.read_csv('startups_profit.csv', index_col=False)
df['State']=df['State'].map({'New York':0,'Florida':1, 'California': 2}).astype(int)

# Training Data
X = df[["R&D Spend", "Administration", "Marketing Spend","State"]]
y = df[["Profit"]]
X, y = df.iloc[:, :-1], df.iloc[:, -1] 

# Setting up train test split
X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), train_size=0.7,random_state=0)
len(X_train), len(X_test), len(y_train), len(y_test)
# Log model to our Project
import mlflow

# Set the connection to the tracking URI
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")
# Set the experiment
mlflow.set_experiment("PotentialStartups")
# Start an MLflow Run
mlflow.start_run()
# Set Autolog for XGBoost
import mlflow.xgboost

mlflow.xgboost.autolog()
# Train our First Model
import xgboost 

xgbr = xgboost.XGBRegressor() 
xgbr.fit(X_train, y_train)
# Evaluate our Model using MLflow. This is Log the metrics for us to MLflow.
eval_data = X_test
eval_data["Profits"] = y_test

# This will load our Model
model_uri = mlflow.get_artifact_uri("model")

# This will run the evaluate Method against our model and our evaluation Data for the Regressor Type.
# Here we are also only selecting the "default" evaluators
result = mlflow.evaluate(
    model_uri,
    eval_data,
    targets="Profits",
    model_type="regressor",
    evaluators="default"
)
# End our Run
mlflow.end_run()
# Run this Cell a few times just to populate some data
import mlflow.xgboost
import xgboost

# Start another MLflow Run
with mlflow.start_run() as run:
    mlflow.xgboost.autolog()

    xgbr = xgboost.XGBRegressor() 
    xgbr.fit(X_train, y_train)

    # Evaluate our Model using MLflow
    eval_data = X_test
    eval_data["Profits"] = y_test
    # This will load our Model
    model_uri = mlflow.get_artifact_uri("model")
    
    # Set the evaluation function
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="Profits",
        model_type="regressor",
        evaluators="default"
    )
    import mlflow
import pandas as pd

# Set Tracking URL 
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")

# Get the Experiment ID
experiment_id = mlflow.get_experiment_by_name("PotentialStartups").experiment_id

# Search runs and output to Pandas DF
evals_df = mlflow.search_runs([experiment_id])
evals_df.info()
# Sort it by r2_score
evals_df = mlflow.search_runs([experiment_id], order_by=["metrics.r2_score DESC"])
evals_df
# Print ONLY the r2_score and the run_id
evals_df[["metrics.r2_score", "run_id"]]
# Create a New Model in The Model Registry using the MLflow Client
import mlflow

# Set out tracking URI
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")

# Create a client connection
client = mlflow.MlflowClient()

# Create a new Model in the Registry called StartupModels
client.create_registered_model("StartupModels")
import mlflow

# SET THESE 2 lines
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")
mlflow.set_experiment("PotentialStartups")
# To begin using the Model Registry, Pick our favorite model from above and register it using the run-id
run_id = "39c71e4e495f452388ca287da575accd"

# Register the model
mlflow.register_model(f"runs:/{run_id}/model", "StartupModels")
#deploy model
import mlflow
# Set the tracking URI
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")
import mlflow

# Notice here we actually use mlflow XGBoost "flavor" to load the model. Check the MLflow Docs for more information on Flavors!
model = mlflow.xgboost.load_model(model_uri="models:/StartupModels/1")
model
# Run a quick Prediction on profit using some fake data

# R&D Spend, Administration, Marketing Spend, State
predict_list = [345349.2, 133337.8, 472345.10, 1]
# Predict
prediction = model.predict([predict_list])
prediction[0]
#automated training
# Lets run a Project using the projects function
import mlflow

# Set our tracking uri
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")

# Run the projects with our specified parameters
mlflow.projects.run(
    # Specifies where the MLproject file lives
    './',
    # Running this on the main entry point
    entry_point='main',
    # Here is our Experiment Name.
    experiment_name='PotentialStartups',
    # Using the local environment
    env_manager='local',
    # Set our Desired parameters for our model
    parameters={
        'n_estimators': 20, 
        'max_depth': 5
    })
# Lets just check to make sure it worked
import mlflow
import pandas as pd

# Set Tracking URL 
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")

# Get the Experiment ID
experiment_id = mlflow.get_experiment_by_name("PotentialStartups").experiment_id
# Search runs and output to Pandas DF. You can get the run_id from the output from the Project run.
evals_df = mlflow.search_runs([experiment_id])
evals_df['run_id']=="7ffd7fbf2cc54685a68756bbf4aeff5c"


