#!/bin/bash

CLUSTER_URL="http://localhost:8088/cluster"
OOZIE_URL="http://localhost:11000/oozie"

WF=$1
oozie_job=`oozie jobs -filter name="$WF" | grep -oi '^[0-9\-]\+[a-z\-]\+ ' | head -n 1`
if [[ -z $oozie_job ]]; then
    echo "Oozie job not found"
    exit
fi

tmp_file=`mktemp /tmp/oozie_monitor.XXXXXX`

while true; do

    oozie job -info $oozie_job > $tmp_file
    cat $tmp_file | sed -e "s/job_\([0-9_]\+\)[^ ]\+/application_\1/g; s/\-\-/\-/g;"
    
    date '+%Y-%m-%d %H:%M:%S'
    echo "Oozie: $OOZIE_URL/?job=$oozie_job"
    
    failed=`grep -E 'FAILED|ERROR|RETRY' $tmp_file | grep -o '^[^ ]\+'`
    if [[ ! -z "${failed//[- ]}" ]]; then
        echo -e "===========================\n"
        oozie job -info $failed > $tmp_file
        job_id=`grep -o 'application_[0-9_]\+ ' $tmp_file`
        cat $tmp_file
        echo "Job: $CLUSTER_URL/app/$job_id"
        break
    fi

    active=`grep -E 'PREP|RUNNING' $tmp_file | grep -o '^[^ ]\+'`
    if [[ -z "${active// }" ]]; then
        echo "Finished =)"
        break
    fi

    sleep 5
done
