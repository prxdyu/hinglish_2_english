# Hinglish to English Translation using Custom Transformer


### Overview
This project implements a custom transformer model for translating text from Hinglish (a mix of Hindi and English) to English. The transformer architecture, known for its effectiveness in sequence-to-sequence tasks, is adapted and trained on a dataset of Hinglish-English parallel sentences to perform translation.

### Features
- Custom implementation of the transformer architecture tailored for Hinglish-English translation.
- Data preprocessing utilities for handling Hinglish text, including tokenization, normalization, and data augmentation.
- Training pipeline with support for automatic checkpoint saving, learning rate scheduling, and evaluation metrics tracking.
- Evaluation function for translating Hinglish sentences to English using the trained model.
- Fine-tuning capabilities for adapting the model to domain-specific or task-specific datasets.


### File Structure

- main.py: Entry point script responsible for importing data, preprocessing, training the transformer model, and performing inference.
- model.py: Implementation of the custom transformer architecture for Hinglish-English translation.
- utils.py: Utility functions for data preprocessing, evaluation, and model training.


### Usage
1. Data Import and Preprocessing: Import Hinglish-English parallel sentences and preprocess them using main.py. Ensure the data is split into training and validation sets.
2. Model Training: Train the custom transformer model by running main.py. Adjust hyperparameters and training settings as needed.
3. Evaluation: Evaluate the trained model's performance on the validation set using the metrics provided by the training script.
4. Inference: Use the trained model to translate Hinglish sentences to English. Input a Hinglish sentence when prompted, and the translated English sentence will be generated.

### Dependencies
- TensorFlow 2.x
- TensorFlow Datasets (for loading and preprocessing data)
- NumPy
- pandas

### Setup Instructions
1. Install the required dependencies using pip install -r requirements.txt.
2. Download or collect a dataset of Hinglish-English parallel sentences and place them in the data directory.
3. Run main.py to preprocess the data, train the model, and perform inference.

### Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
