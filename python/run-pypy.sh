#!/usr/bin/env bash

BINARY=$HOME/bin/pypy

if [ ! -z $(command -v pypy) ]; then
  BINARY=$(command -v pypy)
fi

if [ ! -f $BINARY ]; then
  >&2 echo "Pypy couldnt be found installing it first"
  install_pypy.sh
fi

if [ -f $BINARY ]; then
  EXTRA_PATH=`python3 -c 'from __future__ import print_function; import site; print(":".join(site.getsitepackages()))'`
  BASE_DIR=$( realpath $( dirname -- "${BASH_SOURCE[0]}" )/.. )
  export PYTHONPATH=$PYTHONPATH:$EXTRA_PATH:$BASE_DIR
  exec $BINARY "$@"

else
  >&2 echo "Couldnt load pypy - using python instead"
  exec python3 "$@"

fi

