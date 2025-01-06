import logging
import logging.config
import os
import json
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
        json_collection_uri: str | None = "predictions.json",
        *,
        data_capture: bool = False,
        storage_format: str = "json"
    ) -> None:
        """Initialize the model.

        Args:
            json_collection_uri: Path to JSON file
            data_capture: Whether to capture prediction data
            storage_format: Storage format (only "json" supported)
        """
        self.json_collection_uri = json_collection_uri
        self.data_capture = data_capture
        self.storage_format = storage_format

    def load_context(self, context: PythonModelContext) -> None:
        """Load the transformers and the Keras model specified as artifacts."""
        if not os.getenv("KERAS_BACKEND"):
            os.environ["KERAS_BACKEND"] = "jax"

        import keras

        self._configure_logging()
        logging.info("Loading model context...")

        # Handle JSON URI only
        self.json_collection_uri = os.environ.get(
            "JSON_COLLECTION_URI",
            self.json_collection_uri,
        )

        logging.info("Keras backend: %s", os.environ.get("KERAS_BACKEND"))
        logging.info("JSON collection URI: %s", self.json_collection_uri)

        # Initialize JSON file if it doesn't exist
        self._initialize_storage()

        # Load model components
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])
        self.model = keras.saving.load_model(context.artifacts["model"])

        logging.info("Model is ready to receive requests")

    def _initialize_storage(self):
        """Initialize JSON storage file if it doesn't exist."""
        try:
            if not os.path.exists(self.json_collection_uri):
                with open(self.json_collection_uri, 'w') as f:
                    json.dump([], f)
                logging.info(f"Created new JSON file at {self.json_collection_uri}")
            elif os.path.getsize(self.json_collection_uri) == 0:
                with open(self.json_collection_uri, 'w') as f:
                    json.dump([], f)
                logging.info(f"Initialized empty JSON file at {self.json_collection_uri}")
        except Exception as e:
            logging.error(f"Error initializing JSON file: {str(e)}")

    def predict(self, context: PythonModelContext, model_input, params: dict[str, Any] | None = None) -> list:
        """Make predictions using the model."""
        import pandas as pd
        import numpy as np
        import logging

        try:
            # Add explicit logging for params
            logging.info(f"Received params: {params}")
            
            # FIXED: Properly extract storage format from params or use default
            if params and 'storage_format' in params:
                self.storage_format = params['storage_format']  # Update instance variable
            logging.info(f"Using storage format: {self.storage_format}")
            
            # Extract data capture preference
            should_capture = params.get('data_capture', self.data_capture) if params else self.data_capture
            
            # Log the settings
            logging.info(f"Data capture: {should_capture}, Storage format: {self.storage_format}")
            
            # Handle different input formats
            if isinstance(model_input, dict) and 'inputs' in model_input:
                data = pd.DataFrame(model_input['inputs'])
            elif isinstance(model_input, pd.DataFrame):
                if len(model_input.shape) == 3:
                    data = pd.DataFrame(model_input.values.reshape(model_input.shape[0], -1))
                else:
                    data = model_input
            elif isinstance(model_input, np.ndarray):
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

            # Capture data if enabled
            if should_capture:
                self.capture(data, predictions, self.storage_format)  # Use instance storage_format

            logging.info(f"Predictions: {predictions}")
            return predictions

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}", exc_info=True)
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

    def capture(self, model_input: pd.DataFrame, model_output: list, storage_format: str = "json") -> None:
        """Save the input request and output prediction to JSON only."""
        logging.info("Capturing data in JSON format...")

        # Create a copy and rename columns back
        data = model_input.copy()
        reverse_mapping = {
            'bill_length_mm': 'culmen_length_mm',
            'bill_depth_mm': 'culmen_depth_mm'
        }
        data = data.rename(columns=reverse_mapping)

        # Generate UUID and timestamp
        entry_uuid = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc)

        try:
            # Prepare JSON entry
            json_entry = {
                "timestamp": current_time.isoformat(),
                "uuid": entry_uuid,
                "input": data.to_dict(orient='records')[0],  # Get first record
                "prediction": model_output[0] if model_output else None
            }

            # Read existing JSON data
            json_data = []
            if os.path.exists('predictions.json'):
                with open('predictions.json', 'r') as f:
                    try:
                        json_data = json.load(f)
                    except json.JSONDecodeError:
                        json_data = []
                        logging.warning("Could not decode existing JSON, starting fresh")

            # Append new entry
            json_data.append(json_entry)

            # Write back to file
            with open('predictions.json', 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logging.info("Successfully wrote to predictions.json")

        except Exception as e:
            logging.error(f"Error writing to JSON: {str(e)}")

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