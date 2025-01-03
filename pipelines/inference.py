import logging
import logging.config
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext


class Model(mlflow.pyfunc.PythonModel):
    """A custom model that can be used to make predictions.

    This model implements an inference pipeline with three phases: preprocessing,
    prediction, and postprocessing. The model will optionally store the input requests
    and predictions in a SQLite database.

    The [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc)
    blog post is a great reference to understand how to use custom Python models in
    MLflow.
    """

    def __init__(
        self,
        data_collection_uri: str | None = "penguins.db",
        *,
        data_capture: bool = False,
    ) -> None:
        """Initialize the model.

        By default, the model will not collect the input requests and predictions. This
        behavior can be overwritten on individual requests.

        This constructor expects the connection URI to the storage medium where the data
        will be collected. By default, the data will be stored in a SQLite database
        named "penguins" and located in the root directory from where the model runs.
        You can override the location by using the 'DATA_COLLECTION_URI' environment
        variable.
        """
        self.data_capture = data_capture
        self.data_collection_uri = data_collection_uri
        self.models = []  # Add list to store ensemble models

    def load_context(self, context: PythonModelContext) -> None:
        """Load the transformers and the Keras model specified as artifacts.

        This function is called only once as soon as the model is constructed.
        """
        # By default, we want to use the JAX backend for Keras. You can use a different
        # backend by setting the `KERAS_BACKEND` environment variable.
        if not os.getenv("KERAS_BACKEND"):
            os.environ["KERAS_BACKEND"] = "jax"

        import keras

        self._configure_logging()
        logging.info("Loading model context...")

        # If the DATA_COLLECTION_URI environment variable is set, we should use it
        # to specify the database filename. Otherwise, we'll use the default filename
        # specified when the model was instantiated.
        self.data_collection_uri = os.environ.get(
            "DATA_COLLECTION_URI",
            self.data_collection_uri,
        )

        logging.info("Keras backend: %s", os.environ.get("KERAS_BACKEND"))
        logging.info("Data collection URI: %s", self.data_collection_uri)

        # Load all models from the ensemble
        self.models = []
        for i in range(5):  # Load all 5 cross-validation models
            model_path = context.artifacts[f"model_{i}"]
            model = keras.saving.load_model(model_path)
            self.models.append(model)

        # Load transformers as before
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        logging.info(f"Loaded ensemble of {len(self.models)} models")
        logging.info("Model is ready to receive requests")

    def predict(self, context: PythonModelContext, model_input, params: dict[str, Any] | None = None) -> list:
        """Make predictions using the model."""
        import pandas as pd
        import numpy as np
        import logging

        try:
            # Handle different input formats
            if isinstance(model_input, dict) and 'inputs' in model_input:
                data = pd.DataFrame(model_input['inputs'])
            elif isinstance(model_input, pd.DataFrame):
                if len(model_input.shape) == 3:
                    # Reshape 3D DataFrame to 2D
                    data = pd.DataFrame(model_input.values.reshape(model_input.shape[0], -1))
                else:
                    data = model_input
            elif isinstance(model_input, np.ndarray):
                # Handle numpy array input
                if len(model_input.shape) == 3:
                    data = pd.DataFrame(model_input.reshape(model_input.shape[0], -1))
                else:
                    data = pd.DataFrame(model_input)
            else:
                data = pd.DataFrame([model_input])

            logging.info(f"Input data shape: {data.shape}")
            logging.info(f"Input columns: {data.columns.tolist()}")

            # Ensure we have the expected columns
            expected_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 
                              'body_mass_g', 'sex', 'island']
            
            # If we have numeric indices, map them to expected column names
            if all(isinstance(col, int) for col in data.columns):
                data.columns = expected_columns

            # Rename columns for the transformer
            column_mapping = {
                'culmen_length_mm': 'bill_length_mm',
                'culmen_depth_mm': 'bill_depth_mm'
            }
            data = data.rename(columns=column_mapping)
            
            logging.info(f"Processed columns: {data.columns.tolist()}")

            # Transform features
            transformed_input = self.features_transformer.transform(data)
            transformed_input = np.asarray(transformed_input)
            
            logging.info(f"Transformed input shape: {transformed_input.shape}")

            # Get predictions
            raw_predictions = self.model.predict(transformed_input, verbose=0)
            predicted_classes = np.argmax(raw_predictions, axis=1)
            confidences = np.max(raw_predictions, axis=1)

            # Get species encoder and convert predictions to labels
            species_encoder = self.target_transformer.named_transformers_['species']
            predicted_labels = species_encoder.inverse_transform(
                predicted_classes.reshape(-1, 1)
            ).flatten()

            # Format output
            predictions = [
                {
                    "prediction": str(label),
                    "confidence": float(confidence)
                }
                for label, confidence in zip(predicted_labels, confidences)
            ]

            logging.info(f"Predictions: {predictions}")
            return predictions

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}", exc_info=True)
            logging.error(f"Input type: {type(model_input)}")
            if isinstance(model_input, (np.ndarray, pd.DataFrame)):
                logging.error(f"Input shape: {model_input.shape}")
            logging.error(f"Full error:", exc_info=True)
            return []

    def process_output(self, output: np.ndarray) -> list:
        """Process the prediction received from the model.

        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        logging.info("Processing prediction received from the model...")

        result = []
        if output is not None:
            prediction = np.argmax(output, axis=1)
            confidence = np.max(output, axis=1)

            # Let's transform the prediction index back to the
            # original species. We can use the target transformer
            # to access the list of classes.
            classes = self.target_transformer.named_transformers_[
                "species"
            ].categories_[0]
            prediction = np.vectorize(lambda x: classes[x])(prediction)

            # We can now return the prediction and the confidence from the model.
            # Notice that we need to unwrap the numpy values so we can serialize the
            # output as JSON.
            result = [
                {"prediction": p.item(), "confidence": c.item()}
                for p, c in zip(prediction, confidence, strict=True)
            ]

        return result

    def capture(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save the input request and output prediction to the database.

        This method will save the input request and output prediction to a SQLite
        database. If the database doesn't exist, this function will create it.
        """
        logging.info("Storing input payload and predictions in the database...")

        connection = None
        try:
            connection = sqlite3.connect(self.data_collection_uri)

            # Let's create a copy from the model input so we can modify the DataFrame
            # before storing it in the database.
            data = model_input.copy()

            # We need to add the current time, the prediction and confidence columns
            # to the DataFrame to store everything together.
            data["date"] = datetime.now(timezone.utc)

            # Let's initialize the prediction and confidence columns with None. We'll
            # overwrite them later if the model output is not empty.
            data["prediction"] = None
            data["confidence"] = None

            # Let's also add a column to store the ground truth. This column can be
            # used by the labeling team to provide the actual species for the data.
            data["species"] = None

            # If the model output is not empty, we should update the prediction and
            # confidence columns with the corresponding values.
            if model_output is not None and len(model_output) > 0:
                data["prediction"] = [item["prediction"] for item in model_output]
                data["confidence"] = [item["confidence"] for item in model_output]

            # Let's automatically generate a unique identified for each row in the
            # DataFrame. This will be helpful later when labeling the data.
            data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]

            # Finally, we can save the data to the database.
            data.to_sql("data", connection, if_exists="append", index=False)

        except sqlite3.Error:
            logging.exception(
                "There was an error saving the input request and output prediction "
                "in the database.",
            )
        finally:
            if connection:
                connection.close()

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys
        from pathlib import Path

        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )
