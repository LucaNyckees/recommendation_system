#!/bin/bash
make download
make load
python . src/dashboard/dashboard.py