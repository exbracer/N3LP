#########################################################################
# File Name: perf.sh
# Author: qiao_yuchen
# mail: qiaoyc14@mails.tsinghua.edu.cn
# Created Time: Thu Jul 14 18:00:37 2016
#########################################################################
#!/bin/bash

PERF_DATA_LOCATION=../perf_result/
OUTPUT=$PERF_DATA_LOCATION
OUTPUT+=perf.data

EXEC=$1
INPUT_DIMENTION=50
HIDDEN_DIMENTION=50

DATASET_LOCATION=../n3lp_data/
INPUT=$DATASET_LOCATION
INPUT+=$2

MINI_BATCH_SIZE=$3
NUM_THREADS=$4

echo $INPUT
echo $EXEC
echo $OUTPUT
sudo perf record -o $OUTPUT -g -s -a -- $EXEC -d $INPUT -i $INPUT_DIMENTION -h $HIDDEN_DIMENTION -m $MINI_BATCH_SIZE -n $NUM_THREADS

sudo perf report -i $OUTPUT -m -T -g -C -I -b
