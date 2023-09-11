#!/bin/bash

usage="Usage: `basename $0` <workflow name or job id>

Get info about a running coordinator or job by their ID (ending in -C or -W).
If other string is used instead, it will try to find a matching coodinator by that part of the name, using oozie-find.sh"

if [[ -z $1 || "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi


# Info  by coord ID or name

coord=$1
if [[ "$coord" != *"-W" && "$coord" != *"-C" ]]; then
    echo "Finding coordinator by name $coord with oozie-find.sh..."
    coord=$(oozie-find.sh $coord -id)
    error=$?
    echo "$coord"
    if [ $error != 0 ]; then
       echo "Job not found univocally"
       exit 1
    fi
fi


if [[ "$coord" = *"-W" ]]; then
    oozie job -info $coord -verbose | less -F 
else
    oozie job -info $coord | less -F 
fi
