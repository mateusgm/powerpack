#!/usr/bin/env bash

PROJECT="your_project_here"
BUCKET="gs://your_bucket_here"

ACTION=$1
DATASET=`echo $2 | grep -Po '^\w+'`
TABLE=`echo $2 | grep -Po '\w+$'`
SUFIX=$3


if [[ $ACTION == "download" ]]; then
    hdfs dfs -rm -r $BUCKET/${TABLE}${SUFIX}
    set -e

    bq --project_id=PROJECT \
      --location=EU extract \
      --destination_format=CSV \
      --field_delimiter="\t" \
      "$PROJECT:$DATASET.$TABLE" \
      $BUCKET/${TABLE}${SUFIX}/part-*

    echo "Uploaded to $BUCKET/${TABLE}${SUFIX}"
else
    # upload
    if [[ -z $BQ_ONLY ]]; then
        hdfs dfs -rm -r $BUCKET/${TABLE}${SUFIX}
        cat | hdfs dfs -put - $BUCKET/${TABLE}${SUFIX}
    fi

    # creating table
    echo "DROP TABLE IF EXISTS $DATASET.$TABLE" | bq query --nouse_legacy_sql

    bq --project_id=$PROJECT \
        --location=EU load \
        --source_format=CSV \
        --autodetect \
        "$PROJECT:$DATASET.$TABLE" \
        $BUCKET/${TABLE}${SUFIX}

fi
