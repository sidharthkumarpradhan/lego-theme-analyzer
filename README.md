# 🧱 LEGO Set Analyzer & Theme Predictor

![LEGO Analyzer Banner](logo.svg)

## Overview

LEGO Set Analyzer is an advanced data analysis tool for LEGO enthusiasts that helps you optimize your collection, predict themes from brick assortments, and make smarter purchasing decisions. Built with Streamlit and powered by machine learning, this application provides deep insights into the LEGO universe.

## ✨ Features

### 🔍 Theme Prediction
- Predict the theme of a LEGO collection based on its part composition
- Analyze any set of bricks to determine which official LEGO theme they most closely match
- View key parts that influence theme classification with advanced visualization

### 📊 Part Overlap Analysis
- Discover how parts are shared between sets within themes or across different themes
- Identify which sets share the most common parts through interactive heatmaps
- Export overlap data for further analysis

### 💰 Set Purchase Optimizer
- Find the optimal combination of sets to maximize parts coverage while minimizing cost
- Analyze which sets from your wishlist provide the most unique parts for your collection
- Determine buildability of sets based on your current inventory

### 🧩 Theme Requirements Analysis
- Discover the distinctive parts that define specific LEGO themes
- Identify signature pieces that make themes unique
- Analyze your inventory to see which themes you're ready to build

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required Python packages (see `requirements.txt`)
- Access to [Rebrickable.com](https://rebrickable.com/) data (built-in download functionality)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lego-set-analyzer.git
cd lego-set-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

### Database Setup
- The application can operate with an SQLite database for better performance
- Data will be automatically downloaded from Rebrickable on first use
- A local database will be created in the `lego_data` directory

## 🧠 How It Works

### Theme Prediction
The application uses a machine learning model (Random Forest Classifier with PCA) to analyze brick compositions and predict themes. Principal Component Analysis identifies the most important patterns in parts data, which are then used for classification.

### Part Analysis
The system analyzes part frequency and distribution across sets and themes, identifying common patterns and unique elements that characterize different LEGO themes. The visualization tools make it easy to see how parts are shared between different sets.

### Optimization Algorithms
Set purchase optimization uses specialized algorithms to find the minimal combination of sets that maximize parts coverage, helping you make smarter purchasing decisions.

## 📷 Screenshots

![Theme Prediction](attached_assets/image_1746301416437.png)
*Theme Prediction with detailed part analysis*

![Part Overlap Analysis](attached_assets/image_1746301724514.png)
*Heatmap showing part overlap between sets*

![Set Purchase Optimizer](attached_assets/image_1746384902125.png)
*Optimize your LEGO purchases for maximum coverage*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data provided by [Rebrickable.com](https://rebrickable.com/)
- Built with [Streamlit](https://streamlit.io/)
- Inspired by the global community of LEGO enthusiasts

---

<p align="center">
  Made with ❤️ for LEGO builders everywhere
</p>
