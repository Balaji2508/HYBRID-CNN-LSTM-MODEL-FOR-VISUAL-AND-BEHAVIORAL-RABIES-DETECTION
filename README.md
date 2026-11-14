# AI-Based Rabies Detection System for Dogs

## Overview
This project is an AI-powered system designed to detect rabies in dogs using two approaches:
1.  **Image Classification**: Uses a ResNet50 model to identify visual signs of 'Furious Rabies' from images.
2.  **Text Classification**: Uses an LSTM model to analyze symptom descriptions for detecting 'Dumb/Paralytic Rabies'.

## Features
-   **Dual-Mode Detection**: Video/Image and Text analysis.
-   **Custom Datasets**: Trained on processed images and user-provided symptom descriptions.
-   **Safety First**: Provides immediate safety advice based on detection results.

## Installation
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  Run the application:
    ```bash
    streamlit run app.py
    ```
2.  Select "Image Classification" or "Text Classification" from the sidebar.

## Project Structure
-   `data/`: Contains datasets.
-   `models/`: Saved models (`rabid_classifier.h5`, `text_model.h5`).
-   `image_model/`, `text_model/`: Training scripts.
-   `app.py`: Streamlit application.
