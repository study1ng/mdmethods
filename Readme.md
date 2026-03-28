# mdmethods
A framework for Medical Deep learning methods

## Usage
entry point: python main.py ...
--help-main to check help of main
-mh to check help of a specific method (you should fill all arguments of main.py)

## How to add new experiments
you have two way to add new experiment. 
1. you can write the logic at mdmethods/experiments/(your method name)
2. you can write the logic at mdmethods/experiments/(your method name)/experiments/(your experiment name)
for reproducibility, I recommend you use second way, add logic to mdmethods/experiments/(your method name) and never edit the file of implemented experiment.