import mlflow
import numpy as np

class CustomStackingClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, estimators, final_estimator):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def predict(self, context, model_input):
        # Load the individual models
        xgb_models = {}
        for name, path in self.estimators.items():
            xgb_models[name] = mlflow.xgboost.load_model(path)

        # Load the final estimator
        final_estimator = mlflow.sklearn.load_model(self.final_estimator)
        
        # Make predictions
        predictions = final_estimator.predict_proba(np.stack([xgb_models[x].predict_proba(model_input)[:, 1] for x in xgb_models], axis=1))

        return predictions

class StackingClassifierModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the custom model from the context
        estimators = {
                      'xgb1': context.artifacts["estimator-xgb1"],
                      'xgb2': context.artifacts["estimator-xgb2"],
                      'xgb3': context.artifacts["estimator-xgb3"]
                     }
        final_estimator = context.artifacts["final_estimator"]

        return CustomStackingClassifier(estimators, final_estimator)

    def predict(self, context, model_input):
        model = self.load_context(context)
        return model.predict(context, model_input)
    
import logging
from utils import find_project_path
import shutil
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import StackingClassifier
def train_model():
    project_path = find_project_path()
    mlflow.set_tracking_uri(f'{project_path}/mlruns')
    mlflow.set_experiment('custom-ml')
    with mlflow.start_run(run_name='train_model') as run:
        #Load dataset
        df_train = pd.read_csv(f'{project_path}/dataset/train.csv')
        X_train = df_train.drop(['target'],axis=1)
        y_train = df_train['target']
        
        #Train model
        stack0 = [
                  (
                   'xgb1', XGBClassifier(
                                         booster='gbtree', 
                                         objective='binary:logistic', 
                                         tree_method='hist', 
                                         max_depth=10, 
                                         random_state=61
                                        )
                  ),
                  (
                   'xgb2', XGBClassifier(
                                         booster='gbtree', 
                                         objective='binary:logistic', 
                                         tree_method='hist', 
                                         max_depth=8, 
                                         random_state=61
                                        )
                  ),
                  (
                   'xgb3', XGBClassifier(
                                         booster='gbtree', 
                                         objective='binary:logistic', 
                                         tree_method='hist', 
                                         max_depth=6, 
                                         random_state=61
                                        )
                  )
                 ]
        model0 = StackingClassifier(estimators=stack0, cv=3)
        model = model0.fit(X_train, y_train)
        
        #Save the artifacts
        artifacts = {}
        #Save the xgboost classifier models
        for n, m in model.named_estimators_.items():
            path = f"{project_path}/artifacts/{n}"
            artifacts[f'estimator-{n}'] = path
            mlflow.xgboost.save_model(m, path)
        #Save the final estimator (sklearn model)
        path = f"{project_path}/artifacts/final_estimator"
        artifacts['final_estimator'] = path
        mlflow.sklearn.save_model(model.final_estimator_, path)
        
        #Log the final model
        model_wrapper = StackingClassifierModelWrapper()
        mlflow.pyfunc.log_model(
                                artifact_path='custom-ml',
                                python_model=model_wrapper,
                                artifacts=artifacts,
                                code_path=[f'{project_path}/src/custom_stacking_classifier.py'],
                                conda_env=f'{project_path}/conda.yaml'
                               )
        
        #Delete the artifacts in the root of the project
        shutil.rmtree(f'{project_path}/artifacts')
        
        #Access rund_id and experiment_id
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        
        #End run
        mlflow.end_run()
    
    print(F'RUN ID: {run_id}')
    print(F'EXPERIMENT ID: {experiment_id}')
    return run_id, experiment_id

if __name__ == '__main__':
    train_model()