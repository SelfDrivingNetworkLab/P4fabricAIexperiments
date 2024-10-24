#!/bin/bash
# This bash program conduct iperf3 sessions with multiple concurrency levels and frequencies
# Developed by Anees Al-Najjar at Oak Ridge National Laboratory, TN, USA
# Script Arguments :
#     $1 number of TCP parallel sessions or called concurrency levels
#     $2 number of iperf fequency levels
#     $3 iperf port number
#     $4 the output file
#     $5 iperfserver
#     $6 Bind address
#
#  sudo chmod 774 iperfscript.sh
#  ./iperfscript.sh 3 3 5101 iperfout.txt 172.16.0.2 <172.17.0.1>
#


C=$1
F=$2
iperfport=$3
outfile=$4
iperfserver=$5
bindaddress=$6


for i in $(eval echo "{1..$C}")
 do
   echo -e "\n##########################################################\n"
   echo -e "*************  Number of TCP flows: ${i}********"
   echo -e "\n##########################################################\n"
   echo -e "\n*****      Concurrency level: ${i}      *********\n" >> $outfile
   for j in $(eval echo "{1..$F}")
    do
     echo -e "\nFrequency time : ${j}  TCP flows: ${i}"
     #iperf3 -c $iperfserver --bind $bindaddress --format m --parallel $i --port $iperfport --logfile $outfile
     iperf3 -c $iperfserver --format g --bandwidth G --parallel $i --port $iperfport --logfile $outfile
    done
 done

