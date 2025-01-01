# Test GitHub Actions workflow
import logging
import os
from pathlib import Path
import platform

# Import common functions
from common import (
    PYTHON,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    FlowMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
    configure_logging,
    packages,
)
from inference import Model
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    environment,
    project,
    pypi_base,
    resources,
    step,
)
from metaflow.cards import Image

configure_logging()


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "jax[cpu]",
        "boto3",
        "packaging",
        "mlflow",
        "setuptools",
        "python-dotenv",
        "psutil",
    ),
)
class Training(FlowSpec, FlowMixin):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to predict the species of
    penguins.
    """

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help=(
            "Minimum accuracy threshold required to register the model at the end of "
            "the pipeline. The model will not be registered if its accuracy is below "
            "this threshold."
        ),
        default=0.7,
    )

    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
        },
    )
    @step
    def start(self):
        """Start the Training pipeline."""
        import mlflow

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        self.training_parameters = {
            "epochs": TRAINING_EPOCHS,
            "batch_size": TRAINING_BATCH_SIZE,
        }

        # Start with load_data instead of going directly to cross_validation and transform
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load the data from CSV files."""
        import pandas as pd
        from pathlib import Path
        
        # Get all CSV files in the data directory
        data_dir = Path("data")
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
        # Load and combine all CSV files
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                print(f"Loaded {file} with {len(df)} rows")
                
                # Map common column name variations
                column_mapping = {
                    'culmen_length_mm': 'bill_length_mm',
                    'culmen_depth_mm': 'bill_depth_mm',
                    'flipper_length_mm': 'flipper_length_mm',
                    'body_mass_g': 'body_mass_g',
                    'species': 'species'
                }
                
                # Rename columns if they exist with different names
                df = df.rename(columns={k: v for k, v in column_mapping.items() 
                                      if k in df.columns})
                
                dataframes.append(df)
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        # Combine all dataframes
        if not dataframes:
            raise ValueError("No valid CSV files were loaded")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined dataset has {len(combined_df)} rows")
        
        # Drop duplicates if any
        combined_df = combined_df.drop_duplicates()
        print(f"After removing duplicates: {len(combined_df)} rows")
        
        # Print available columns for debugging
        print("Available columns:", combined_df.columns.tolist())
        
        # Verify required columns exist
        required_columns = ["species", "bill_length_mm", "bill_depth_mm", 
                           "flipper_length_mm", "body_mass_g"]
        
        missing_columns = [col for col in required_columns 
                          if col not in combined_df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Available columns: {combined_df.columns.tolist()}"
            )
        
        # Store the data
        self.data = combined_df
        
        # Now branch to cross_validation and transform
        self.next(self.cross_validation, self.transform)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold

        # We are going to use a 5-fold cross-validation process to evaluate the model,
        # so let's set it up. We'll shuffle the data before splitting it into batches.
        kfold = KFold(n_splits=5, shuffle=True)

        # We can now generate the indices to split the dataset into training and test
        # sets. This will return a tuple with the fold number and the training and test
        # indices for each of 5 folds.
        self.folds = list(enumerate(kfold.split(self.data)))

        # We want to transform the data and train a model using each fold, so we'll use
        # `foreach` to run every cross-validation iteration in parallel. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        # Let's start by unpacking the indices representing the training and test data
        # for the current fold. We computed these values in the previous step and passed
        # them as the input to this step.
        self.fold, (self.train_indices, self.test_indices) = self.input

        logging.info("Transforming fold %d...", self.fold)

        # We need to turn the target column into a shape that the Scikit-Learn
        # pipeline understands.
        species = self.data.species.to_numpy().reshape(-1, 1)

        # We can now build the SciKit-Learn pipeline to process the target column,
        # fit it to the training data and transform both the training and test data.
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(
            species[self.train_indices],
        )
        self.y_test = target_transformer.transform(
            species[self.test_indices],
        )

        # Finally, let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(
            self.data.iloc[self.train_indices],
        )
        self.x_test = features_transformer.transform(
            self.data.iloc[self.test_indices],
        )

        # After processing the data and storing it as artifacts in the flow, we want
        # to train a model.
        self.next(self.train_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @resources(memory=4096)
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        logging.info("Training fold %d...", self.fold)

        # Let's track the training process under the same experiment we started at the
        # beginning of the flow. Since we are running cross-validation, we can create
        # a nested run for each fold to keep track of each separate model individually.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model from this fold.
            self.mlflow_fold_run_id = run.info.run_id

            # Let's configure the autologging for the training process. Since we are
            # training the model corresponding to one of the folds, we won't log the
            # model itself.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the training data. Notice how we are
            # using the training data we processed and stored as artifacts in the
            # `transform` step.
            self.model = build_model(self.x_train.shape[1])
            self.model.fit(
                self.x_train,
                self.y_train,
                verbose=0,
                **self.training_parameters,
            )

        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data for this fold.
        """
        import mlflow

        logging.info("Evaluating fold %d...", self.fold)

        # Let's evaluate the model using the test data we processed and stored as
        # artifacts during the `transform` step.
        self.loss, self.accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=2,
        )

        logging.info(
            "Fold %d - loss: %f - accuracy: %f",
            self.fold,
            self.loss,
            self.accuracy,
        )

        # Let's log everything under the same nested run we created when training the
        # current fold's model.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics(
                {
                    "test_loss": self.loss,
                    "test_accuracy": self.accuracy,
                },
            )

        # When we finish evaluating every fold in the cross-validation process, we want
        # to evaluate the overall performance of the model by averaging the scores from
        # each fold.
        self.next(self.evaluate_model)

    @card(type="blank")
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "5",
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "300",
            "MLFLOW_TRACKING_INSECURE_TLS": "true"
        },
    )
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process."""
        import mlflow
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        from io import BytesIO
        import psutil
        import platform
        import sys
        import time
        from mlflow.exceptions import MlflowException
        
        # Merge existing artifacts
        self.merge_artifacts(
            inputs, 
            include=["mlflow_run_id", "mlflow_tracking_uri"]
        )

        # Calculate metrics as before
        metrics = [[i.accuracy, i.loss] for i in inputs]
        self.accuracy, self.loss = np.mean(metrics, axis=0)
        self.accuracy_std, self.loss_std = np.std(metrics, axis=0)

        # Create confusion matrix from all folds
        y_true = []
        y_pred = []
        for inp in inputs:
            true_labels = np.argmax(inp.y_test, axis=1)
            pred_labels = np.argmax(inp.model.predict(inp.x_test, verbose=0), axis=1)
            y_true.extend(true_labels)
            y_pred.extend(pred_labels)

        # Define class labels manually since we know them
        class_labels = ['Adelie', 'Chinstrap', 'Gentoo']
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add labels
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add title and axis labels
        ax.set_title('Confusion Matrix - Cross Validation Results')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Add text annotations
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center", color="black")
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Convert plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Save plot for the card
        current.card.append(
            Image(buf.getvalue()),  # Pass bytes instead of figure
            "### Model Evaluation Results\n"
            f"- **Accuracy**: {self.accuracy:.3f} ±{self.accuracy_std:.3f}\n"
            f"- **Loss**: {self.loss:.3f} ±{self.loss_std:.3f}"
        )
        plt.close()

        # Log to MLflow with retry logic
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                with mlflow.start_run(run_id=self.mlflow_run_id):
                    # Log metrics
                    mlflow.log_metrics({
                        "cross_validation_accuracy": self.accuracy,
                        "cross_validation_accuracy_std": self.accuracy_std,
                        "cross_validation_loss": self.loss,
                        "cross_validation_loss_std": self.loss_std,
                    })

                    # Log system metrics
                    mlflow.log_metrics({
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage_percent": psutil.disk_usage('/').percent,
                    })

                    # Log system parameters
                    mlflow.log_params({
                        "python_version": sys.version,
                        "platform": platform.platform(),
                        "processor": platform.processor(),
                        "cpu_count": psutil.cpu_count(),
                        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                        "machine": platform.machine(),
                        "os": platform.system(),
                    })

                    # Try to log GPU info if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            mlflow.log_params({
                                "gpu_name": torch.cuda.get_device_name(0),
                                "gpu_count": torch.cuda.device_count(),
                            })
                    except ImportError:
                        pass

                # If we get here, logging was successful
                break
                
            except MlflowException as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"Warning: Failed to log to MLflow after {max_retries} attempts. Error: {e}")
                    # Continue execution without MLflow logging
                else:
                    print(f"MLflow logging attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        self.next(self.register_model)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        This function transforms the columns of the entire dataset because we'll
        use all of the data to train the final model.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline to process the target column and use it
        # to transform the data.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(
            self.data.species.to_numpy().reshape(-1, 1),
        )

        # Let's build the SciKit-Learn pipeline to process the feature columns and use
        # it to transform the training.
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train_model)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true"
        }
    )
    @step
    def train_model(self):
        """Train the model that will be deployed to production.

        This function will train the model using the entire dataset.
        """
        import mlflow
        import sys
        from pathlib import Path
        import psutil
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Enable autologging without the invalid parameters
            mlflow.autolog(log_models=False)

            # Log initial system state
            mlflow.log_metrics({
                "initial_cpu_percent": psutil.cpu_percent(),
                "initial_memory_percent": psutil.virtual_memory().percent,
            })

            # Let's disable the automatic logging of models during training so we
            # can log the model manually during the registration step.
            mlflow.autolog(log_models=False)

            # Log system metrics using built-in modules
            mlflow.log_params({
                "os_name": os.name,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "cpu_count": os.cpu_count(),
            })

            # Log the pipeline source code
            pipeline_path = Path(__file__)
            mlflow.log_artifact(pipeline_path, artifact_path="source_code")

            # Let's now build and fit the model on the entire dataset.
            self.model = build_model(self.x.shape[1])
            self.model.fit(
                self.x,
                self.y,
                verbose=2,
                **self.training_parameters,
            )

            # Let's log the training parameters we used to train the model.
            mlflow.log_params(self.training_parameters)

            # Log final system state
            mlflow.log_metrics({
                "final_cpu_percent": psutil.cpu_percent(),
                "final_memory_percent": psutil.virtual_memory().percent,
            })

        # After we finish training the model, we want to register it.
        self.next(self.register_model)

    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def register_model(self, inputs):
        """Register the model in the Model Registry.

        This function will prepare and register the final model in the Model Registry.
        This will be the model that we trained using the entire dataset.

        We'll only register the model if its accuracy is above a predefined threshold.
        """
        import tempfile

        import mlflow

        # Since this is a join step, we need to merge the artifacts from the incoming
        # branches to make them available here.
        self.merge_artifacts(inputs)

        # We only want to register the model if its accuracy is above the threshold
        # specified by the `accuracy_threshold` parameter.
        if self.accuracy >= self.accuracy_threshold:
            logging.info("Registering model...")

            # We'll register the model under the experiment we started at the beginning
            # of the flow. We also need to create a temporary directory to store the
            # model artifacts.
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                # We can now register the model using the name "penguins" in the Model
                # Registry. This will automatically create a new version of the model.
                mlflow.pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    registered_model_name="penguins",
                    artifact_path="model",
                    code_paths=[(Path(__file__).parent / "inference.py").as_posix()],
                    artifacts=self._get_model_artifacts(directory),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    # Our model expects a Python dictionary, so we want to save the
                    # input example directly as it is by setting`example_no_conversion`
                    # to `True`.
                    example_no_conversion=True,
                )
        else:
            logging.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold "
                "(%.2f). Skipping model registration.",
                self.accuracy,
                self.accuracy_threshold,
            )

        # Let's now move to the final step of the pipeline.
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        # Let's start by saving the model inside the supplied directory.
        model_path = (Path(directory) / "model.keras").as_posix()
        self.model.save(model_path)

        # We also want to save the Scikit-Learn transformers so we can package them
        # with the model and use them during inference.
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_signature(self):
        """Return the model's signature.

        The signature defines the expected format for model inputs and outputs. This
        definition serves as a uniform interface for appropriate and accurate use of
        a model.
        """
        from mlflow.models import infer_signature

        return infer_signature(
            model_input={
                "island": "Biscoe",
                "culmen_length_mm": 48.6,
                "culmen_depth_mm": 16.0,
                "flipper_length_mm": 230.0,
                "body_mass_g": 5800.0,
                "sex": "MALE",
            },
            model_output={"prediction": "Adelie", "confidence": 0.90},
            params={"data_capture": False},
        )

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}"
            for package, version in packages(
                "scikit-learn",
                "pandas",
                "numpy",
                "keras",
                "jax[cpu]",
            ).items()
        ]


if __name__ == "__main__":
    Training()
