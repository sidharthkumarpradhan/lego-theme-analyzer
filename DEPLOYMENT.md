# Deploying LEGO Set Analyzer to Streamlit Cloud

This guide explains how to deploy your LEGO Set Analyzer application to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. Your LEGO Set Analyzer code uploaded to a GitHub repository
3. A Streamlit Cloud account (free at [streamlit.io/cloud](https://streamlit.io/cloud))

## Step 1: Prepare Your GitHub Repository

1. Make sure your repository includes all necessary files:
   - All Python code files (app.py, data_processor.py, etc.)
   - .streamlit/config.toml
   - requirements.txt (create this from dependencies.md if needed)
   - README.md
   - All other necessary files like logo.svg, etc.

2. Ensure your main application file is named `app.py` (or update the Streamlit deployment settings accordingly).

3. Commit and push all files to GitHub.

## Step 2: Create a requirements.txt File

If not already present, create a requirements.txt file in your repository with the following packages:

```
streamlit==1.30.0
pandas==2.0.0
numpy==1.24.0
plotly==5.14.0
scikit-learn==1.3.0
matplotlib==3.7.0
requests==2.28.0
```

You can use the provided `requirements.txt.sample` file as a template - just rename it to `requirements.txt`.

## Step 3: Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and log in with your GitHub account.

2. Click on "New app" to create a new Streamlit app.

3. Connect to the GitHub repository containing your LEGO Set Analyzer.

4. Configure your app:
   - Set the main Python file path to "app.py"
   - Choose the branch you want to deploy (usually "main" or "master")
   - Keep the default Python version (3.9+)

5. Click "Deploy".

6. Wait while Streamlit Cloud builds and deploys your app. This typically takes a few minutes.

7. Once deployed, you can access your app via the provided URL.

## Step 4: Database Considerations

Since Streamlit Cloud doesn't persist data between sessions, users of your app will need to:

1. Load data fresh on their first visit
2. Use the "Use SQLite Database" option to create a local database
3. Understand that their personal inventory will not persist between sessions unless they export/import their data

## Step 5: Advanced Settings (Optional)

### Custom Domain

1. In the Streamlit Cloud dashboard, go to your app settings.
2. Under "Custom domain", you can set up a subdomain like `lego-analyzer.streamlit.app`.

### Health Checks and Scaling

1. Streamlit Cloud handles scaling automatically.
2. The platform also implements health checks to ensure your app stays responsive.

## Troubleshooting

If your deployment fails:

1. Check the build logs in the Streamlit Cloud dashboard.
2. Ensure all required packages are listed in requirements.txt.
3. Make sure your app runs locally without errors.
4. Check that your app doesn't exceed memory limits (use memory-efficient options where possible).

## Support

For any issues related to Streamlit Cloud specifically, refer to [Streamlit's documentation](https://docs.streamlit.io/) or reach out to their support team.