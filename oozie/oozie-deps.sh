#!/bin/bash

if [[ -z "$1" || "$@" =~ --help ]]; then
    echo  "Usage: $(basename $0) [workflow_directory] (check|debug)"
    echo ""
    echo  "Example: $(basename $0) path/to/wf"
    echo  "         That will print the dependecies of the workflows, recursively."
    echo ""
    echo  "Example: $(basename $0) path/to/wf check"
    echo  "         That will also check that the latest flag is present, taking some naive assumptions on what date to look for."
    exit 1
fi

oozie=$(dirname `realpath $1`) 
wf="$1"

do_debug=""
if [ "$2" == "debug" ]; then
    do_debug="$2"
fi
if [ "$2" == "check" ]; then
    do_check="$2"
fi

depth=4

# Find what workflow writes a given flag. Lots of assumptions are taken here.
function findWorkflow () {

    key="$1"

    f1=$(echo $key| cut -d'/' -f 1)
    f2=$(echo $key| cut -d'/' -f 2)

    match=""
    file=""
    dir=""
    name="$f1"

    # flags are specified in a flagsBaseDir variable, or created explicitly with a mkdir command:

    if [[ $f2 == *"\$"* ]]; then
        match="$(grep -srh $f1 $oozie/* | grep -v uri-template | grep "\(flagsBaseDir\|mkdir\|my \)" | sed -e "s/\"/./g" -e 's/\$/./g' -e 's/=/./g')"
    else
        name="$f1/$f2"
        match="$(grep -srh $name $oozie/* | grep -v uri-template | grep "\(flagsBaseDir\|mkdir\|my \)" | sed -e "s/\"/./g" -e 's/\$/./g' -e 's/=/./g')"
    fi

    # Some workflows were created only to write flags, exclude those:
    if [ ! -z "$match" ]; then
        file=$(grep -slr "$match$" $oozie/* | grep -v "write-flag")
    fi
    if [ ! -z "$file" ]; then
        dir=$(dirname $file)
    fi

    echo "name=\"$name\"; dir=\"$dir\"; debug=\"f1: $f1  f2: $f2 name: $name match: $match file: $file\""


}

# Indent output with some spaces
function tabs() {
     tabs=""
     for i in `seq 1 $1`;
        do
           tabs="$tabs  "
        done  
     echo -e "$tabs"
}

# Flag patterns contain date keywords: replace them with some actual values.
function replaceTokens() {

    YEAR=`date +"%Y"`
    MONTH=`date +"%m"`
    DAY=`date +"%d"`

    dep=$(echo $1 | sed -e 's/${HOUR}/00/' -e 's/${DAY}/'$DAY'/' -e 's/${MONTH}/'$MONTH'/' -e 's/${YEAR}/'$YEAR'/' -e 's/\/OK//') 
    echo $dep
}

# Extract the flag name from the whole hdfs path. Here, some assuptions.
function getFlagPath() {

    if [[ $1 == *"flags"* ]]; then
        echo $1 | sed -e 's/.*\/flags\///g'
    else
        echo $1 | sed -e 's/\/hive_tables\/raw_tables\///g'
    fi
}

function getWorkflowPath() {
    echo ${1#${GIT_TREE}/}
}

# Take the dependency flags mentioned in the current directory's coordinator.xml, and find in what other workflows they were written:
function findDependencies () {

    cd $1 
    if [ ! -f coordinator.xml ]; then
        echo "no coordinator.xml in the current directory"
        exit 1;
    fi


    for flag in $(cat coordinator.xml | xpath -q -e '//uri-template/text()') 
    do
        key=$(getFlagPath $flag)

        eval $(findWorkflow $key)

        if [ ! -z "$do_debug" ]; then
            echo "$flag  -> $key"
            echo $debug
        fi

        repFlag=$(replaceTokens $flag)

        report="$2 $(tabs $2)Dependency: $name ($repFlag)"

        if [ ! -z "$do_check" ]; then
            present=$(hdfs dfs -ls $repFlag | grep "OK")
            if [ -z "$present" ]; then
                report="$report -> MISSING"
            else 
                report="$report -> OK"
            fi
        fi

        echo "$report"

        if [ ! -z "$dir" ]; then
            echo "$2   $(tabs $2)Workflow: $(getWorkflowPath $dir)"
            if [ $2 -lt $depth ]; then
                nlevel=$(($2 + 1))
                findDependencies $dir $nlevel
            fi
        fi

    done

}

# Start dumping the tree:
(
  cd $oozie
  echo "0 Workflow: $(getWorkflowPath $wf)"
  findDependencies $wf 0
)
