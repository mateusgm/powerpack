#!/usr/bin/env bash

usage="usage: $(basename $0) [vowpal parameters]
Run vw with --audit and make the output more readable."

if [[ "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi


vw "${@:1}" --audit \
    | sed -E 's/^([^\t])/------\n\1/g; s/\t+/\n/g'
