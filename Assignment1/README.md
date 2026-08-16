# Custom C++ Deep Learning Framework

## Overview
This repository contains a custom Deep Learning framework implemented in C++ (backend) and Python (frontend).

## Prerequisites
* C++ Compiler (GCC/Clang/MSVC) with OpenMP support (it is usually)
* CMake (3.18+)
* Python 3.8+ (we used 3.11)
* `pybind11` (Install via `pip install pybind11`)

## 1. Build Instructions
The C++ backend must be compiled before running any Python scripts.

We assume python venv in this (Assignment1) folder. After that

```bash
# 1. Create build directory
mkdir build
cd build

# 2. Configure CMake
cmake ..

# 3. Build
cmake --build . --config Release


Note: The compiled shared library (mytorch.cpython...so or .pyd) will be automatically placed in the python/ directory.

2. Dataset Setup
Please place the dataset binary files in the following locations:

Dataset 1 (Digits): python/data_1

Dataset 2 (Objects): python/data_2

Training command 
Dataset 1:
python python/train_cnn.py --data_dir python/data_1 --config python/config_d1.json --save_path python/model_d1.txt
Dataset 2:
python python/train_cnn.py --data_dir python/data_2 --config python/config_d2.json --save_path python/model_d2.txt


Evaluate Dataset 1
python python/eval.py --data_dir python/data_1 --model_path python/model_d1.txt

Evaluate Dataset 2
python python/eval.py --data_dir python/data_2 --model_path python/model_d2.txt

Files
cpp/: C++ source code 
python/: Python scripts.
GNR638Project.pdf:  project report and analysis.

