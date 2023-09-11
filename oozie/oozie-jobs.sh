#!/bin/bash

usage="`basename $0` <coordinator name> [-id]

Find running coordinators and workflows by part of their names.

If -id is used, and there's only one match, that ID is returned as a single string."

if [[ -z "$1" || "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi

error=$?
if [ $error != 0 ]; then
    echo "First run kinit-me"
    exit 1
fi


name=$1
if [ -z "$1" ]; then
    echo "We need a worflow name"
    exit 1
fi

function find_with_string()  {
  oozie jobs -jobtype coordinator -filter status="RUNNING" -len 10000 | grep $name
  oozie jobs -filter status="RUNNING" -len 10000 | grep $name
}

hits=$(find_with_string)
exact=`echo "$hits" | grep -- "-C" | grep " ${name}RUNNING"`


# Line breaks are gone here, find how to keep them TODO TODO TODO
if [ "$exact" != '' ]; then
  # The string was found exactly as one coofdinator name, not just as a part of it.
  result="$exact"
else
  result="$hits"
fi

if [ "$2" = "-id" ]; then
    coord=$(echo "$result" | cut -f 1 -d" ")
    if [ -z "$coord" -o $(echo "$coord" | wc -l) != 1 ]; then
        >&2 echo "`echo "$coord" | wc -l` results found"
        exit 1
    fi
    echo $coord
else
    echo "$result"
fi


