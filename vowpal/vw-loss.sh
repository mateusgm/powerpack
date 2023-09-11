#!/usr/bin/env bash

usage="usage: $(basename $0) [vowpal parameters]
Run vw and outputs only the average loss."

if [[ "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi

vw "${@:1}" 2>&1 \
    | grep 'average loss' \
    | grep -oP '[\d\.]+'
