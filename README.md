Here’s an updated **README.md** file that aligns with your `pyproject.toml` and highlights the dependencies installed via Poetry:

---

# Recurrent Neural Networks Project

This project contains three Jupyter notebooks that cover both the theoretical and practical aspects of Recurrent Neural Networks (RNNs), including their variants such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU). The notebooks provide a comprehensive understanding of these models, with practical applications in time series prediction and sentiment analysis.

## Notebooks Overview

### 1. **Recurrent Neural Networks.ipynb**
This notebook provides a detailed theoretical overview of RNNs and their advanced variants, LSTM and GRU. The following topics are covered:
- **Recurrent Neural Networks (RNNs)**: 
  - Introduction to RNNs and their general architecture.
  - Explanation of how RNNs process sequential data.
- **Long Short-Term Memory (LSTM)**:
  - In-depth discussion of LSTM's architecture, including input, forget, and output gates.
  - Equations that describe the LSTM's internal workings.
- **Gated Recurrent Unit (GRU)**:
  - Overview of GRU’s simpler structure with reset and update gates.
  - Comparison of GRU and LSTM architectures.
- **Comparison of RNN, LSTM, and GRU**:
  - Comparison table summarizing the differences in architecture, complexity, performance, and use cases.

### 2. **TimeSeries_prediction_using_RNNs.ipynb**
This notebook focuses on applying RNN models to time series data. It demonstrates how RNNs can be used for predicting energy consumption using real-world data:
- **Dataset**: The notebook uses a dataset of energy power consumption in the USA.
- **Data Preprocessing**: Prepares the data for time series analysis.
- **Model Implementation**: 
  - RNN, LSTM, and GRU models are implemented.
  - Training and evaluation of models for predicting future energy consumption.
- **Performance Comparison**: Compares the performance of RNN, LSTM, and GRU on this dataset.

### 3. **Sentiment_Analysis_using_LSTM.ipynb**
This notebook demonstrates the use of LSTM for a Natural Language Processing (NLP) task, specifically sentiment analysis:
- **Dataset**: A sentiment analysis dataset (e.g., movie reviews or social media posts) is used to classify the sentiment (positive/negative).
- **LSTM Model for Sentiment Analysis**: 
  - Detailed implementation of LSTM for text classification.
  - Text preprocessing using tokenization and embedding layers.
  - Training and evaluation of the LSTM model for sentiment prediction.
  
---

## How to Run the Project

1. **Install Dependencies using Poetry**:
   This project uses [Poetry](https://python-poetry.org/) for dependency management. The dependencies are defined in the `pyproject.toml` file. To install the necessary packages, ensure Poetry is installed and run the following command:
   ```bash
   poetry install
   ```

   The key dependencies are:
   - `numpy` `<2`
   - `pandas` `^2.2.2`
   - `torch`, `torchvision`, and `torchaudio` (PyTorch with CUDA 12.1 support)
   - `torch-scatter` and `torch-cluster`
   - `matplotlib`, `seaborn`, and `nltk` for plotting and NLP tasks
   - `scikit-learn` for machine learning utilities
   - `notebook` for running Jupyter notebooks

2. **Clone the Repository**:
   Clone the project repository from GitHub:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   ```

3. **Run the Notebooks**:
   Activate the virtual environment created by Poetry and launch Jupyter notebooks:
   ```bash
   poetry shell
   jupyter notebook
   ```

4. **Open the Notebooks**:
   Open the relevant notebooks (`Recurrent Neural Networks.ipynb`, `TimeSeries_prediction_using_RNNs.ipynb`, `Sentiment_Analysis_using_LSTM.ipynb`) and run the cells to explore the theory and implementations.

---

## Project Structure

```bash
.
├── Recurrent Neural Networks.ipynb
├── TimeSeries_prediction_using_RNNs.ipynb
├── Sentiment_Analysis_using_LSTM.ipynb
├── README.md
└── pyproject.toml
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- The energy consumption dataset is provided by [source].
- The sentiment analysis dataset is taken from [source].
- Special thanks to the contributors and open-source libraries used in this project.

---
