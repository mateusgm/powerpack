#!/usr/bin/env bash

usage="usage: $(basename $0) [vowpal parameters]

Run vw with the given parameters and log the result to log.txt so
you can have a history of things you've tried."

if [[ "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi


vw "${@:1}" 2>&1 \
  | tee >( ( ggrep 'average loss' | ggrep -oP '[\d\.]+' | tr '\n' '\t'; echo "${@:1}" ) >> log.txt )
