# Recommendation System for Amazon Products with enhanced BERT classifier

## Description
We download Amazon product reviews data from https://amazon-reviews-2023.github.io/, load them to a postgres database and develop a recommender system.
- embed reviews using a pre-trained embedding BERT or TF-IDF method
- build a sentiment classifier (compare various models amongst random forests, gradient boosting and BERT)
- create an interactive dashboard for data visualization and model performance analysis
- implement the recommendation system
- wrap up the app using FastAPI routes and dockerinzing it

## People
- Luca Nyckees
- Silvio Innocenti-Malini

## Pipeline Snapshot
<img src="https://github.com/LucaNyckees/recommendation_system/blob/main/images/rs_pipeline.png?raw=true" width="700">


## Basic structure
```
├── LICENSE
|
├── config files (.env, .ini, ...)
|
├── README.md
│
├── docs/               
│
├── requirements.txt  
|
├── __main__.py
│
├── src/                
|     ├── __init__.py
|     └── _version.py
|
└── tests/
```

### Wrapped up worflow with Docker
First, make sure you have *Docker* installed on your machine. If you wish to make all the steps yourself, without using Docker, you can go to the next section.

Run the following command.
```
make build
```
This will create (~5min) a Docker image named `recommendation_system:latest` with the virtual environment in which the app will run.

Once this is done, run 
```
make run
```
This will launch a dashboard at the address localhost:8000 : have a look !

### Step-by-step workflow

Let's say your project directory looks like `ROOT_DIR := PRE_ROOT_DIR / recommendation_system`.

0. Setup a Postgres database and specify your associated credentials in the `config.toml` file.

1. Create a virtual environment named `venv` and install dependencies and requirements in it
```
make venv
```
2. Activate your virtual environment.
```
source venv/bin/activate
```
3. Download Amazon datasets into your project (run from `PRE_ROOT_DIR`).
```
python recommendation_system download datasets
```
4. Load downloaded datasets to your Postgres database (run from `PRE_ROOT_DIR`).
```
python recommendation_system load datasets
```
5. Launch the FastAPI application - it creates routes that will be called by the dashboard (run from `ROOT_DIR`).
```
uvicorn src.fastapi_app.main:app
```
6. Launch the Dash dashboard have check the result at `localhost:8000` in your favorite browser (run from `ROOT_DIR`).
```
python src/dashboard/dashboard.py
```
