#!/bin/bash

#pingscript.sh <count> <output file> <server>
count=$1
output=$2
server=$3

ping -c$1 $3 > $2
