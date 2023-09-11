#!/usr/bin/env bash

usage="usage: $(basename $0)

Apply the tag trick on a stream of vowpal input data. Example:
   cat data.vw | vw_tagging.sh | vw ..."

if [[ "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi

awk -v q="'" '{
  tag = q$1" ";
  i   = ( index($2,q) || index($2,"|") ) ? 2 : 3;
  $i  = split($i,a,"|") > 1 ? tag"|"a[2] : tag;
  print $0;
}'


