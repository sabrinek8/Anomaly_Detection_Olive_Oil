# 🫒 Anomaly Detection in Olive Oil Quality

This project applies **deep learning (Autoencoders)** to detect anomalies in olive oil quality data. It combines **exploratory data analysis (EDA)**, preprocessing, and a **neural autoencoder model** to classify normal vs. anomalous oil quality.

## 📂 Project Structure

- `anomaly_detection_olive_oil.ipynb` – Main notebook with preprocessing, training, and evaluation.


### Main Libraries

- `pandas`, `numpy` – data handling
- `matplotlib` – visualization
- `scikit-learn` – preprocessing, metrics
- `tensorflow / keras` – building the autoencoder model

## 📊 Workflow

1. **Exploratory Data Analysis (EDA)**
   - Load dataset, check distributions, missing values, and correlations.
   - Feature engineering (transformations, categorical encoding, normalization).

2. **Preprocessing**
   - One-hot encoding of categorical variables
   - Standardization of numerical features
   - Train/test split

3. **Modeling**
   - Build an **Autoencoder neural network** with Keras
   - Train on "normal" samples
   - Use reconstruction error to detect anomalies

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Confusion Matrix & PR/F1 analysis
   - Visualization of anomalies vs. normal samples

## 🚀 Usage

Run the Jupyter Notebook step by step:

```bash
jupyter notebook anomaly_detection_olive_oil.ipynb
```

Or convert the pipeline into a Python script if needed.

## 📈 Results

- The autoencoder successfully identifies abnormal olive oil samples with high recall.
- Reconstruction error thresholding provides robust separation of anomalies.
- Visualizations (confusion matrix, error distributions) demonstrate anomaly detection performance.

## 🔮 Future Improvements

- Test with different anomaly detection methods (Isolation Forest, One-Class SVM).
- Perform hyperparameter tuning for autoencoder depth and regularization.
- Deploy as a **FastAPI** or **Flask API** for real-time anomaly detection.
