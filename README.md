# Title

## Description

## People

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
