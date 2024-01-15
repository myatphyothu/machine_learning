#!/bin/bash

export FLASK_APP=./service/index.py
flask --debug run -h 0.0.0.0 -p 80