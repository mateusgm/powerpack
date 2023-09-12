#!/bin/bash

usage="usage: parallel.sh 'script to be called'

parallel.sh speed up consumption a stream of data by calling a script in parallel.

Examples:
  cat dataset.csv       | parallel.sh ./my_parser.py -a arg1 -b arg2
  cat dataset.gz        | parallel.sh ./my_parser.py -a arg1 -b arg2
  echo /folder/of/files | parallel.sh ./my_parser.py -a arg1 -b arg2

By default, it splits the data into blocks of 50M using 50% of parallelism with a max of 10 processes.
To change the behaviour:

  cat dataset.csv | B=100M parallel.sh ./my_parser.py
  cat dataset.csv | T=20   parallel.sh ./my_parser.py
"
if [[ "$@" =~ --help ]]; then
  echo "$usage" && exit 
fi


# parameters

BLOCK="${B:-50M}"
BATCH="${N}"
THREADS=${T:-$((`nproc`-2))}
PARAMS=${P:-"--lb"}
SCRIPT=${@:1}
TMPDIR=/tmp

MAX_THREADS=10
if [[ -z $T ]] && [[ $THREADS > $MAX_THREADS ]]; then
    THREADS=$MAX_THREADS
fi

if [[ ! -z $BATCH ]]; then
  PARAMS="$PARAMS -N $BATCH"
fi


# speed ups

if [[ $USE_PYPY == "1" ]]; then
  TOKENS=(${SCRIPT// / })
  BINARY=`command -v ${TOKENS[0]}`
  SCRIPT="run-pypy.sh ${BINARY:-${TOKENS[0]}} ${TOKENS[@]:1}" 
fi

if [[ -d /mnt/disks/local_ssd/ ]]; then
    export TMPDIR=/mnt/disks/local_ssd/tmp
    mkdir -p $TMPDIR
    PARAMS="--tmpdir $TMPDIR $PARAMS"
fi


# composing the command

info() {
    if [[ $V == 1 ]]; then
        >&2 echo "$@"
    fi
}

if [[ $SCRIPT == *{}* ]]; then
    info ">> Reading params from stdin"

    STREAM_CMD="cat $FIRST_LINE -"

else
    # detecting input

    IFS= read -r input_sample
    IFS= read -r input_sample2

    MAYBE_FILE=`echo -e "$input_sample" | cut -f1 -d' ' | head -n 1`
    FIRST_LINE=`mktemp $TMPDIR/parallel.XXXXXX`

    printf '%s\n' "$input_sample" > $FIRST_LINE
    if [[ ! -z $input_sample2 ]]; then
        printf '%s\n' "$input_sample2" >> $FIRST_LINE
        MORE_LINES=1
    fi

    # detecting compression

    if [[ `echo $MAYBE_FILE/*`  == *"gz"* ]]; then
      info ">> Gzip detected"
      DECOMPRESS_CMD="zcat"

    elif [[ `echo $MAYBE_FILE/*` == *"bz"* ]]; then
      info ">> Bzip detected"
      DECOMPRESS_CMD="bzcat"
    fi


    # composing command

    if [[ -d $MAYBE_FILE ]]; then
        info ">> Reading from list of folders"

        CAT_CMD="$DECOMPRESS_CMD {} |"
        STREAM_CMD="cat $FIRST_LINE - | xargs -I{} find {} -type f"

    elif [[ -f $MAYBE_FILE ]] && [[ $MORE_LINES == 1 ]]; then
        info ">> Reading from list of files"

        CAT_CMD="$DECOMPRESS_CMD {} |"
        STREAM_CMD="cat $FIRST_LINE -"

    elif [[ -f $MAYBE_FILE ]]; then

        HEADER=`mktemp /tmp/parallel.XXXXXX`
        ${DECOMPRESS_CMD:-cat} `cat $FIRST_LINE` | head -n 1 > $HEADER

        if [[ -z $DECOMPRESS_CMD ]]; then
            info ">> Reading from text file"

            PARAMS="$PARAMS --pipepart --block $BLOCK -a $MAYBE_FILE"
            CAT_CMD="cat $HEADER - |"
            STREAM_CMD="cat"

        else
            info ">> Reading from compressed file"

            PARAMS="$PARAMS --pipe --block $BLOCK"
            CAT_CMD="cat $HEADER - |"
            STREAM_CMD="$DECOMPRESS_CMD $MAYBE_FILE"
        fi

    else
        info ">> Reading pipe from stdin"

        PARAMS="$PARAMS --pipe --block $BLOCK"
        CAT_CMD="cat $FIRST_LINE - |"
        STREAM_CMD="cat"

    fi

fi
    


# running

info
eval $STREAM_CMD \
    | parallel \
        --will-cite \
        -j$THREADS \
        $PARAMS \
        "$CAT_CMD $SCRIPT"

trap "killall parallel; exit 1" INT

