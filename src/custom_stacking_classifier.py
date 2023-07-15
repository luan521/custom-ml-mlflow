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