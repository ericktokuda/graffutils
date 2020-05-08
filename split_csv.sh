#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "  $0 <csv-path> <number of partitions>"
	exit 0
fi

N=$(wc -l < $1)
L=$((N/$2))
echo $K

H=$(head -n1 $1)

for I in $(seq 0 $(($2-1)));
do
	FILEIN=$(basename $1)
	EXT="${1##*.}"
	F="${FILEIN%.*}-part${I}.csv"
	M=$((L*I))
	C=$((N-M-1))
	echo "$H" > /tmp/$F
	tail "$1" -n "$C" | head -n "$L" >> /tmp/$F
done
