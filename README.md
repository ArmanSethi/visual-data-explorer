# 🔍 Visual Data Explorer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

An AI/ML project for **searchable, interactive visualizations** of heterogeneous datasets scraped live from the internet. This project enables automated data collection, cleaning, embedding generation, clustering, and interactive exploration through a modern web dashboard.

## 🌟 Features

- **🌐 Automated Web Scraping**: Collect images, tables, and text from websites
- **🧼 Data Cleaning & Normalization**: Process and standardize heterogeneous data
- **🧠 Embedding Generation**: Create semantic embeddings for images and text using pre-trained models
- **📈 Clustering & Analysis**: Group similar items using KMeans, DBSCAN, or Hierarchical clustering
- **🖥️ Interactive Dashboard**: Explore data through a Streamlit-powered web interface
- **🔍 Semantic Search**: Find relevant items using text or image queries
- **🤖 GitHub Actions**: Automated retraining and repository updates

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modules](#modules)
- [Dashboard](#dashboard)
- [GitHub Actions](#github-actions)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/ArmanSethi/visual-data-explorer.git
cd visual-data-explorer
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Scrape Data

```python
from scraper.web_scraper import WebScraper

scraper = WebScraper("https://example.com")
images = scraper.scrape_images("https://example.com/gallery")
tables = scraper.scrape_tables("https://example.com/data")
text_data = scraper.scrape_text("https://example.com/article")
```

### Generate Embeddings

```python
from embeddings.embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator(model_name='clip')
generator.load_models()

image_embeddings = generator.generate_image_embeddings(image_paths)
text_embeddings = generator.generate_text_embeddings(texts)
```

### Cluster Data

```python
from embeddings.clustering import ClusteringEngine

engine = ClusteringEngine(method='kmeans', n_clusters=5)
labels = engine.fit(embeddings)
engine.visualize_clusters(embeddings, labels)
```

## 📂 Project Structure

```
visual-data-explorer/
├── scraper/              # Web scraping modules
│   ├── __init__.py
│   └── web_scraper.py    # Main scraping functionality
├── processing/          # Data cleaning and normalization
│   └── data_cleaner.py   # Data cleaning utilities
├── embeddings/          # Embedding generation and clustering
│   ├── embedding_generator.py  # Generate embeddings
│   └── clustering.py     # Clustering algorithms
├── dashboard/           # Streamlit web application
│   └── app.py            # Main dashboard application
├── data/                # Data storage
│   ├── raw/             # Raw scraped data
│   ├── processed/       # Cleaned data
│   └── embeddings/      # Generated embeddings
├── docs/                # Documentation
├── .github/
│   └── workflows/       # GitHub Actions workflows
│       └── retrain.yml   # Automated retraining
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 💻 Usage

### Command Line Interface

#### Web Scraping

```bash
python -m scraper.web_scraper --url https://example.com --output data/raw
```

#### Data Cleaning

```bash
python -m processing.data_cleaner --input data/raw --output data/processed
```

#### Generate Embeddings

```bash
python -m embeddings.embedding_generator --input data/processed --output data/embeddings
```

#### Run Clustering

```bash
python -m embeddings.clustering --embeddings data/embeddings --method kmeans --n-clusters 5
```

## 🧱 Modules

### Scraper Module

**Location**: `scraper/web_scraper.py`

Handles automated web scraping for images, tables, and text content.

**Key Features**:
- Multi-format data extraction (images, tables, text)
- Configurable scraping parameters
- Rate limiting and error handling
- Data persistence

### Processing Module

**Location**: `processing/data_cleaner.py`

Cleans and normalizes heterogeneous data.

**Key Features**:
- Text cleaning and normalization
- Table data cleaning (duplicates, missing values)
- Image normalization (size, format)
- Outlier detection and removal

### Embeddings Module

**Location**: `embeddings/`

Generates embeddings and performs clustering.

**Key Features**:
- Multi-modal embedding generation (CLIP-based)
- Similarity search
- Multiple clustering algorithms
- Dimensionality reduction for visualization

### Dashboard Module

**Location**: `dashboard/app.py`

Interactive Streamlit dashboard for data exploration.

**Key Features**:
- Data upload and visualization
- Interactive clustering controls
- Semantic search interface
- Real-time data exploration

## 🖥️ Dashboard

The dashboard provides an intuitive interface for exploring your data:

1. **Overview Tab**: Dataset statistics and cluster distributions
2. **Images Tab**: Image gallery with upload functionality
3. **Text Tab**: Text document viewer and analyzer
4. **Search Tab**: Semantic search across all data types

### Features:

- 📁 **File Upload**: Upload images, text files, and CSV data
- 📊 **Visualization**: Interactive cluster visualizations
- 🔍 **Search**: Text and image-based semantic search
- ⚙️ **Configurable**: Adjust clustering parameters in real-time

## 🤖 GitHub Actions

Automated workflows for continuous integration and model updates.

### Retrain Workflow

**Location**: `.github/workflows/retrain.yml`

Automatically:
- Scrapes new data weekly
- Cleans and processes data
- Regenerates embeddings
- Updates cluster assignments
- Commits changes to repository

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
SCRAPER_USER_AGENT=Mozilla/5.0...
EMBEDDING_MODEL=clip
CLUSTERING_METHOD=kmeans
N_CLUSTERS=5
```

### Model Configuration

Edit `config.yaml` to customize:
- Scraping parameters
- Embedding model settings
- Clustering algorithms
- Dashboard preferences

## 📊 Sample Datasets

Sample datasets are provided in `data/samples/`:

- `sample_images/`: 50 sample images
- `sample_text/`: Text documents
- `sample_tables/`: CSV data files

## 📝 Documentation

Detailed documentation is available in the `docs/` directory:

- `API.md`: API reference
- `ARCHITECTURE.md`: System architecture
- `DEPLOYMENT.md`: Deployment guide
- `TUTORIAL.md`: Step-by-step tutorials

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 Roadmap

- [ ] Add support for video data
- [ ] Implement real-time scraping
- [ ] Add more embedding models (BERT, ResNet)
- [ ] Enhanced search with filtering
- [ ] User authentication and data privacy
- [ ] API endpoints for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment support

## 📞 Contact

**Author**: Arman Sethi
**GitHub**: [@ArmanSethi](https://github.com/ArmanSethi)
**Repository**: [visual-data-explorer](https://github.com/ArmanSethi/visual-data-explorer)

## 🙏 Acknowledgments

- OpenAI CLIP for embeddings
- Streamlit for the dashboard framework
- scikit-learn for clustering algorithms
- Beautiful Soup for web scraping

---

**Built with ❤️ for AI/ML enthusiasts**
