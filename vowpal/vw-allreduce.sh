#!/usr/bin/env bash

usage="usage: $(basename $0) <db.input_table> </hdfs/path/of/output> [vowpal parameters]

Run vowpal wabbit in Allreduce mode (i.e. training is distributed on hadoop cluster).

Example:
  $(basename $0) db.table hdfs:///path/to/output_dir \\
    -b 28 -q aa --loss_function logistic cat -l 0.01"

if [[ "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi


# hadoop streaming settings

JOBS_HDFS="hdfs:///jobs.txt"
JOBS_TMP="/tmp/jobs.txt"

MAPPER_MEMORY=${MAPPER_MEMORY:-8192}    # memory per mapper
MAPPER_CORES=${MAPPER_CORES:-2}         # cores per mapper
STREAM_THREADS=${STREAM_THREADS:-2}     # number of processes acessing hdfs
STREAM_BATCH=${STREAM_BATCH:-10}        # number of files per hdfs process
STREAM_TIMEOUT=${STREAM_TIMEOUT:-"30m"} # timeout for hdfs streaming


# modes

MODEL_NAME=`echo "${@:1}" | grep -Po '\-f \K[^ ]+'`

if [[ $1 == "mapred" ]]; then
    INPUT=$2
    OUTPUT=$3
    ARGS="${@:4}"

    # creating jobs file

    hdfs dfs -rm -r $OUTPUT $JOBS_HDFS 2> /dev/null
    hdfs dfs -ls -C $INPUT | grep -v '_SUCCESS' > $JOBS_TMP
    hdfs dfs -put $JOBS_TMP $JOBS_HDFS

    JOB_CNT=`cat $JOBS_TMP | wc -l`
    MAPPERS=${MAPPERS:-$JOB_CNT}
    LINES_MAP=$((JOB_CNT/$MAPPERS))

    MOD=$((JOB_CNT%$MAPPERS))
    if (( $MOD != 0)); then
        LINES_MAP=$(( LINES_MAP + 1))
    fi

    # hadoop streaming

    hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar \
        -D yarn.resourcemanager.scheduler.monitor.enable=false \
        -D mapreduce.task.timeout=600000000 \
        -D mapreduce.job.reduce=0 \
        -D mapreduce.map.memory.mb=$MAPPER_MEMORY \
        -D mapreduce.map.cpu.vcores=$MAPPER_CORES \
        -D mapreduce.input.lineinputformat.linespermap=$LINES_MAP \
        -files $0 \
        -inputformat  org.apache.hadoop.mapred.lib.NLineInputFormat \
        -input $JOBS_HDFS \
        -output $OUTPUT \
        -mapper "`basename $0` mapper $ARGS" \
        -reducer NONE

    if [[ ! -z $MODEL_NAME ]]; then
        rm -f $MODEL_NAME*
        hdfs dfs -get $OUTPUT/$MODEL_NAME* .
    fi

    # -verbose
    # -D mapred.max.maps.per.node=1 \
    # -D mapred.tasktracker.map.tasks.maximum
    # -D mapreduce.input.fileinputformat.split.minsize

elif [[ $1 == "report" ]]; then
    job_cnt=`cat /tmp/jobs.txt | wc -l`
    all_red="--total $((job_cnt+1)) --node 0 --span_server localhost --unique_id 0"

    vw_args="${@:2} $all_red -k -c -f model.dat"
    hdfs dfs -cat `head -n 1 /tmp/jobs.txt` | vw $vw_args


elif [[ $1 == "mapper" ]]; then
    # get settings from environment variables set by Hadoop Streaming

    mapper=`printenv mapred_task_id | cut -d "_" -f 5`
    unique_id=`printenv mapred_job_id | tr -d 'job_'`
    node=`expr 0 + $mapper`
    span_server=$mapreduce_job_submithostname
    output_dir=$mapreduce_output_fileoutputformat_outputdir

    # getting partitions

    grep -Po '\K[^\s]+$' > partitions.txt
    date +"[%F | %T] Data: `cat partitions.txt | wc -l` files" > /dev/stderr

    if [[ `cat partitions.txt | grep '.gz'` ]]; then
       DECOMPRESS="| zcat"
    fi

    if [[ `cat partitions.txt | grep '.bz'` ]]; then
       DECOMPRESS="| bzcat"
    fi

    # running vw

    all_red="--total $mapreduce_job_maps --node $node --span_server $span_server --unique_id $unique_id"
    vw_args=`echo " ${@:2} " | sed -e "s/ -k / /g; s/ -c / -k --cache_file $node.cache --compressed /g"`
    date +"[%F | %T] Starting: vw $vw_args $all_red" > /dev/stderr

    cat partitions.txt \
        | timeout $STREAM_TIMEOUT \
          parallel -j$STREAM_THREADS -N$STREAM_BATCH --lb "hdfs dfs -cat {} $DECOMPRESS" \
        | vw $vw_args $all_red 1>&2

    # wrapping up

    if [[ ! -z $MODEL_NAME ]] && [[ $node == "0" ]]; then
        date +"[%F | %T] Finished: storing model" > /dev/stderr
        hadoop fs -put $MODEL_NAME* $output_dir/
    fi

    date +"[%F | %T] Done node=$node" > /dev/stderr


elif [[ $1 == "spanning" ]]; then
    killall spanning_tree 2> /dev/null
    spanning_tree > spanning.log 2> /dev/null
    tail -f spanning.log | awk '{ count++; print count" | "$0 }'


else # master
    trap "killall spanning_tree tail 2> /dev/null" 0 1 2 3 6

    $0 spanning &
    $0 mapred "${@:1}"


fi
