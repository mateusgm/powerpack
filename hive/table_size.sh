#!/usr/bin/env bash

_PATH=$1
LAST_K=${2:-9999999}
ROWS=${3:-10000}


# table or file?

if ! [[ $_PATH =~ .*/.* ]]; then 
    TABLE=`echo $_PATH | sed -e 's/\./.db\//g'`
    _PATH="/user/hive/warehouse/$TABLE/"
fi


# getting data

sample=`hadoop fs -ls $_PATH/* | grep -o '/.*' | grep -v 'SUCCESS' | tail -n 1` 
s_rows=`hadoop fs -cat $sample 2> /dev/null | head -n $ROWS | wc -m`

c_files=`hadoop fs -ls $_PATH/* | tail -n $LAST_K | wc -l`
c_parts=`hadoop fs -ls $_PATH   | tail -n $LAST_K | wc -l`
s_total=`hadoop fs -ls $_PATH/* | tail -n $LAST_K | awk '{ s+=$5 } END { print s }'`


# printing data

printf "Size\t%.2f Gb\n" $(echo "$s_total/1024/1024/1024" | bc -l)
printf "Rows\t%d M\n"    $((s_total/s_rows/100))
printf "Files\t%d\n"     $c_files

echo "* based on $ROWS rows and $c_parts partitions"

