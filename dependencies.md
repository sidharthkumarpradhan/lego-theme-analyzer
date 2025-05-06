# LEGO Set Analyzer Dependencies

This application requires the following Python packages:

```
streamlit==1.30.0
pandas==2.0.0
numpy==1.24.0
plotly==5.14.0
scikit-learn==1.3.0
matplotlib==3.7.0
requests==2.28.0
```

## Installation

When deploying the app, ensure these dependencies are properly installed. If deploying on Streamlit sharing, they will be automatically installed from the `requirements.txt` file.

For local development:

```bash
pip install streamlit pandas numpy plotly scikit-learn matplotlib requests
```

## Creating requirements.txt

Create a `requirements.txt` file with the exact same content as above before deploying to Streamlit Cloud. Using exact version numbers helps ensure reproducibility and avoids compatibility issues.