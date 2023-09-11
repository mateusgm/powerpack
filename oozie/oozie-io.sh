#!/bin/bash
set -e

usage="usage: oozie-io.sh workflow-path

List all input and output tables of an oozie workflow."

if [[ -z $1 || "$@" =~ --help ]]; then
  echo "$usage" && exit
fi


DIR=$1
ESC=`echo $DIR | sed -E 's/\//\\\\\//g'`

echo -e "---- Input ----"
grep -oRP --color=always '(FROM|JOIN) ([^ (]+)' $DIR/* \
  | sed -E 's/FROM|JOIN//g' \
  | sed -E "s/$ESC//g" \
  | uniq -u

echo -e "\n---- Output ----"
grep -zoRP --color=always 'TABLE( IF( NOT)* EXISTS)*\s+[^\s]+' $DIR/* \
  | perl -pe 's/(TABLE|EXISTS)\n/TABLE /g' \
  | sed -E 's/(:.*)( .*)/:\2/g; s/;//g;' \
  | sed -E "s/$ESC//g" \
  | uniq -u

echo

