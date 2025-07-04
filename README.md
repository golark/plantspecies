---
title: Plantspeciesdetector
emoji: 🦀
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Plant Species Detector & Care Assistant

A machine learning application that detects plant species from uploaded images using a trained fastai model, enhanced with **RAG (Retrieval-Augmented Generation)** for intelligent plant care recommendations.

## Features

- Upload plant images (JPG, JPEG, PNG)
- Real-time plant species detection
- Confidence score display
- **RAG-enhanced plant care tips** using ChromaDB vector database
- **Interactive care questions** - ask specific plant care questions
- **Dual search modes** - upload images OR search by plant name
- **Semantic search** for similar plants and care information
- User-friendly web interface with modern UI

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Deployment Options

### 1. Streamlit Cloud (Recommended - Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Select your repository
5. Set the main file path to `streamlit_app.py`
6. Deploy!

### 2. Heroku

1. Create a `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git add .
git commit -m "Initial commit"
git push heroku main
```

### 3. Railway

1. Connect your GitHub repository to Railway
2. Set the start command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
3. Deploy automatically

### 4. Google Cloud Platform

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

2. Deploy to Google Cloud Run:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/plant-detector
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/plant-detector --platform managed
```

## Model Information

The app uses a pre-trained fastai model (`plant_species_model.pkl`) that can classify various plant species including:
- Aloe Vera, Azalea, Bamboo, Begonia, Cactus
- Camellia, Carnation, Cherry Blossom, Chrysanthemum
- Daffodil, Daisy, Fern, Gardenia, Geranium
- Hibiscus, Hydrangea, Ivy, Jasmine, Lavender
- Lily, Magnolia, Maple, Marigold, Moss
- Oak, Orchid, Palm, Peony, Petunia
- Pine, Poppy, Rhododendron, Rose, Snapdragon
- Sunflower, Tulip, Violet, Zinnia

## Requirements

- Python 3.9+
- fastai 2.7.19
- streamlit 1.46.1
- torch 2.6.0
- PIL/Pillow
- numpy

## License

MIT License
