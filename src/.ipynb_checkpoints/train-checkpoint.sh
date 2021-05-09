#!/bin/bash
script_path1="/home/kaan.aytekin/Thesis/src/cross_validation_LM.py"
log_path1="/home/kaan.aytekin/Thesis/logs/cross_validation_LM.log"

script_path2="/home/kaan.aytekin/Thesis/src/cross_validation_RF.py"
log_path2="/home/kaan.aytekin/Thesis/logs/cross_validation_RF.log"

script_path3="/home/kaan.aytekin/Thesis/src/cross_validation_XGB.py"
log_path3="/home/kaan.aytekin/Thesis/logs/cross_validation_XGB.log"

source /home/kaan.aytekin/Thesis/thesis_environment/bin/activate

start_script(){
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "Started $1 $dt" | tee -a $2
    python -u $1 >>$2 2>&1
    dt=$(date '+%d/%m/%Y %H:%M:%S')
    echo "Ended $1 $dt" | tee -a $2
}

#start_script $script_path1 $log_path1 &
#start_script $script_path2 $log_path2 &
start_script $script_path3 $log_path3 &
wait
