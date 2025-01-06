import logging
import logging.config
import os
import sys
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
        data_collection_uri: str | None = "predictions.json",
        *,
        data_capture: bool = False,
        storage_format: str = "json"
    ) -> None:
        """Initialize the model.

        Args:
            data_collection_uri: Path to JSON file
            data_capture: Whether to capture prediction data
            storage_format: Storage format (only "json" supported)
        """
        self.data_collection_uri = data_collection_uri
        self.data_capture = data_capture
        self.storage_format = storage_format

    def load_context(self, context: PythonModelContext) -> None:
        """Load the transformers and the Keras model specified as artifacts."""
        # Set Keras backend to JAX before importing keras
        os.environ["KERAS_BACKEND"] = "jax"
        
        # Force keras to reinitialize with the new backend
        if 'keras' in sys.modules:
            del sys.modules['keras']
        
        import keras

        self._configure_logging()
        logging.info("Loading model context...")

        # Handle JSON URI only
        self.data_collection_uri = os.environ.get(
            "DATA_COLLECTION_URI",
            self.data_collection_uri,
        )

        logging.info("Keras backend: %s", os.environ.get("KERAS_BACKEND"))
        logging.info("Data collection URI: %s", self.data_collection_uri)

        # Initialize storage file if data capture is enabled
        if self.data_capture:
            self._initialize_storage()

        # Load model components
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])
        self.model = keras.saving.load_model(context.artifacts["model"])

        logging.info("Model is ready to receive requests")

    def _initialize_storage(self):
        """Initialize JSON storage file if data capture is enabled."""
        # Only initialize storage if data capture is enabled
        if not self.data_capture:
            return
        
        try:
            if not os.path.exists(self.data_collection_uri):
                with open(self.data_collection_uri, 'w') as f:
                    json.dump([], f)
                logging.info(f"Created new JSON file at {self.data_collection_uri}")
            elif os.path.getsize(self.data_collection_uri) == 0:
                with open(self.data_collection_uri, 'w') as f:
                    json.dump([], f)
                logging.info(f"Initialized empty JSON file at {self.data_collection_uri}")
        except Exception as e:
            logging.error(f"Error initializing JSON file: {str(e)}")

    def process_input(self, input_data: pd.DataFrame) -> np.ndarray:
        """Process the input data before making predictions.
        
        Args:
            input_data: Input DataFrame to process
            
        Returns:
            Processed numpy array ready for model prediction
        """
        try:
            logging.info(f"Original input data:\n{input_data}")
            logging.info(f"Original columns: {input_data.columns.tolist()}")

            # Make a copy of the input data to avoid modifying the original
            processed_data = input_data.copy()

            # Only rename columns if they exist in the input
            if 'culmen_length_mm' in processed_data.columns:
                # Rename columns for the transformer (from culmen to bill)
                column_mapping = {
                    'culmen_length_mm': 'bill_length_mm',
                    'culmen_depth_mm': 'bill_depth_mm'
                }
                processed_data = processed_data.rename(columns=column_mapping)
                
                logging.info(f"After renaming columns: {processed_data.columns.tolist()}")
                logging.info(f"Transformed data:\n{processed_data}")

            # Transform features using the original input for test cases
            # and processed data for real predictions
            if 'culmen_length_mm' in input_data.columns:
                transformed_input = self.features_transformer.transform(processed_data)
            else:
                transformed_input = self.features_transformer.transform(input_data)
                
            logging.info(f"After feature transformation shape: {transformed_input.shape}")
            
            return transformed_input

        except Exception as e:
            logging.error(f"Error processing input: {str(e)}", exc_info=True)
            logging.error(f"Input data that caused error:\n{input_data}")
            return None

    def predict(self, context: PythonModelContext, model_input, params: dict[str, Any] | None = None) -> list:
        """Make predictions using the model."""
        try:
            # Extract data capture preference from params or use default
            should_capture = params.get('data_capture', self.data_capture) if params else self.data_capture
            
            # Handle different input formats
            if isinstance(model_input, list):
                data = pd.DataFrame(model_input)
            elif isinstance(model_input, dict) and 'inputs' in model_input:
                data = pd.DataFrame(model_input['inputs'])
            elif isinstance(model_input, pd.DataFrame):
                data = model_input
            elif isinstance(model_input, np.ndarray):
                data = pd.DataFrame(model_input)
            else:
                data = pd.DataFrame([model_input])

            # Use process_input for feature transformation
            transformed_input = self.process_input(data)
            if transformed_input is None:
                logging.error("Input processing failed")
                return []

            # Get predictions
            raw_predictions = self.model.predict(transformed_input, verbose=0)
            
            # Process the output
            predictions = self.process_output(raw_predictions)

            # Capture data if enabled
            if should_capture:
                self.capture(data, predictions, self.storage_format)

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
        """Save the input request and output prediction."""
        logging.info("Capturing data in JSON format...")

        try:
            # Generate UUID and timestamp
            entry_uuid = str(uuid.uuid4())
            current_time = datetime.now(timezone.utc)

            # Prepare JSON entries for all samples
            json_entries = []
            for input_record, prediction in zip(model_input.to_dict(orient='records'), model_output):
                json_entries.append({
                    "timestamp": current_time.isoformat(),
                    "uuid": str(uuid.uuid4()),
                    "input": input_record,
                    "prediction": prediction
                })

            # Read existing JSON data
            json_data = []
            if os.path.exists(self.data_collection_uri):
                with open(self.data_collection_uri, 'r') as f:
                    try:
                        json_data = json.load(f)
                    except json.JSONDecodeError:
                        json_data = []
                        logging.warning("Could not decode existing JSON, starting fresh")

            # Append new entries
            json_data.extend(json_entries)

            # Write back to file
            with open(self.data_collection_uri, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logging.info(f"Successfully wrote {len(json_entries)} entries to {self.data_collection_uri}")

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