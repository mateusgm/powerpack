#!/bin/bash

if [[ -z "$1" || "$@" =~ --help ]]; then
    echo "Usage: `basename $0` workflow/path [action]
    
    Action can be:
    - dir: Find the flags directory of a given oozie
    - flags: List the flags written by a workflow
    - missing: Make a summary of gaps in daily workflow flags"
    exit
fi

if [[ -z "$1" || "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi



function findDir() {
    wf_path=`realpath $1`

    grep "\(flagsBaseDir\|mkdir\|my \|flags_base_dir\)" $wf_path/job.properties \
      | sed  -e 's/ //g' -e 's/flagsBaseDir=//' -e 's/flags_base_dir=//'
}

function listFlags() {
    dir=$(findDir $1) 
    echo "Listing: $dir"

    hdfs dfs -ls $dir | less -F 
}

function missingFlags() {
    listFlags $1 |
        grep -Eo  "/[0-9]{4}-[0-9]{2}-[0-9]{2}" |   ## extract date pattern
        tr -d '/' |                             ## remove leading /
        sort |                                  ## ensure the dates are in order
        date -f - '+%s' |                       ## convert dates into epoch
        ## find gaps longer than one day (3600 * 24 seconds)
        ## and print out the first missing date and missing number of days
        awk '{if(NR>1 && ($1 - _n) > 90000) {print strftime("%Y-%m-%d", _n + 3600*24) " " ($1 - _n)/(3600*24) - 1};_n=$1}'
}

if [[ $2 == "dir" ]]; then
    findDir $1
fi

if [[ $2 == "list" ]]; then
    listFlags $1
fi

if [[ $2 == "missing" ]]; then
    missingFlags $1
fi

