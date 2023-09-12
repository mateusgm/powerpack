#!/usr/bin/env bash
# 
# usage:
#   vw-hogwild.sh /tmp/dataset <mapper> <concurrency> <vowpal params>
#

DATA_DIR=$1
MAPPER=$2
THREADS=$3

if [[ ! -d $DATA_DIR ]]; then
    echo "Error: dataset should be a directory"
    echo "This script is intended to parse a partitioned dataset in parallel"
fi

vw_params="${@:3}"
dataset=`echo $DATA_DIR/* | grep .`


# multiple passes: repeat dataset k times
# TODO: maybe use --holdout_after to get a proper measure?

passes=`echo $vw_params | grep -Po 'passes \K(\d+)'`
passes=${passes:-1}

vw_params=`echo $vw_params | sed -E 's/ --passes [[:digit:]]+//'`
dataset=`printf "$dataset %.0s" $(seq 1 $passes)`


# wrapping cmds
CAT_CMD="cat"
if [[ `echo "$dataset" | grep 'gz'` ]]; then
  CAT_CMD="zcat"
fi

MAP_CMD="| $MAPPER"
if [[ $MAPPER = "cat" ]]; then
  MAP_CMD=""
fi

# cleaning up previous instances

wrap_up() { \
  trap - INT QUIT TERM
  pkill $1 -f 'vw.*--port 26542'
}
wrap_up -9


# vowpal daemon

vw $vw_params \
  --daemon --port 26542 \
  --num_children $THREADS &
trap 'wrap_up -9; exit 1' INT QUIT TERM


# waiting for vw start
while ! nc localhost 26542 </dev/null; do sleep 1; done



# sending training instances on vw way 
parallel -j$THREADS \
  "$CAT_CMD {} $MAP_CMD | nc localhost 26542 > /dev/null" \
  ::: $dataset

# wrap up
wrap_up
