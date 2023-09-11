#!/bin/bash
set -e

usage="usage: oozie-rerun.sh <workflow id> [<skip node 1[, skip node 2,..]]

Rerun an workflow skipping nodes. Example:
  oozie-rerun.sh 9919924-190111132020557-oozie-oozi-W preprocess,train"

if [[ "$@" =~ --help ]]; then
  echo "$usage" && exit
fi

oozie job -Doozie.wf.rerun.skip.nodes=$2 -rerun $1
