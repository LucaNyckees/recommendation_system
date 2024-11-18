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
## Virtual environment
Use the following command lines to create and use venv python package:
```
python3.10 -m venv venv
```
Then use the following to activate the environment:
```
source venv/bin/activate
```
You can now use pip to install any packages you need for the project and run python scripts, usually through a `requirements.txt`:
```
python -m pip install -r requirements.txt
```
When you are finished, you can stop the environment by running:
```
deactivate
```

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

### FastAPI App
Go to the file `src/fastapi_app/README.md` for instructions.

### Dash App
There is an interative dashboard built with Dash, that calls routes from the FastAPI app.
Go to the root directory `PATH_TO_YOUR_PROJECT` and run the following to launch the dashboard.
```
python src/dashboard/dashboard.py
```
There are three sections, namely
- data visualization
- models performance
- recommender demo