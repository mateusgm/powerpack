#!/usr/bin/env bash

unset http_proxy
unset https_proxy

SLEEP=2
SEARCH=$1
URL="${CLUSTER:-http://localhost:8088/cluster/scheduler}"
GRID="%20s %8s %10s %10s\n"

clear
while true; do
    tput cup 0 0
    printf "$GRID" "NAME" "vCPUs" "MEMORY" "PROGRESS"

    wget -O - -o /dev/null $URL \
        | grep $SEARCH \
        | sed -e 's/"//g' \
        | awk -F',' '{ if( $12 ) printf("'"$GRID"'", substr($3,0,16), $12, int($13/1000) " Gb", substr($14,12,4) " %") }'

    echo -e "\n\n"
    sleep $SLEEP
done
