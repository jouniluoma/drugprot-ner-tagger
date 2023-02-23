#!/bin/bash

# from Devlin et al. 2018 (https://arxiv.org/pdf/1810.04805.pdf), Sec. A.3
# """
# [...] we found the following range of possible values to work well across all tasks:
# * Batch size: 16, 32
# * Learning rate (Adam): 5e-5, 3e-5, 2e-5
# * Number of epochs: 2, 3, 4
# """

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
JOBDIR="$DIR/../jobs"

MAX_JOBS=100
BATCH_SIZE=256
INFERENCE_SIZE=100000

mkdir -p "$JOBDIR"
MODELDIR="your model here"
DATADIR="data to .tsv file"
echo "Model: $MODELDIR"
echo "Batch size: $BATCH_SIZE"
echo "Inference size: $INFERENCE_SIZE"

for filename in $DATADIR/*.tsv; do
	while true; do
	jobs=$(ls "$JOBDIR" | wc -l)
	if [ $jobs -lt $MAX_JOBS ]; then break; fi
	echo "Too many jobs ($jobs), sleeping ..."
	sleep 60
	done
	echo "Submitting $filename"
	job_id=$(
	sbatch "$DIR/run-tagger.sh" \
		$BATCH_SIZE \
		$INFERENCE_SIZE \
		$filename \
		$MODELDIR \
		| perl -pe 's/Submitted batch job //'
	)
	echo "Submitted batch job $job_id"
	touch "$JOBDIR"/$job_id
	sleep 10
done
