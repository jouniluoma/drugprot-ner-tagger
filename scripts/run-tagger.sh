#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -p gputest
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001426
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


function on_exit {
    rm -f out-$SLURM_JOBID.tsv
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT


batch_size=$1
sentences_on_batch=$2
input_file=$3
ner_model=$4

echo "Used NER model: $ner_model"
echo "batch size: $batch_size"
echo "Samples per inference: $sentences_on_batch"
echo "data file: $train_file"

rm -f logs/latest.out logs/latest.err
ln -s "$SLURM_JOBID.out" "logs/latest.out"
ln -s "$SLURM_JOBID.err" "logs/latest.err"


module purge
module load tensorflow/2.8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"


srun singularity_wrapper exec python3 tagger.py \
    --vocab_file "vocab" \
    --bert_config_file "config" \
    --init_checkpoint "model" \
    --learning_rate 1 \
    --num_train_epochs 2 \
    --max_seq_length 128 \
    --batch_size $batch_size \
    --train_data "$input_file" \
    --test_data "$test_file" \
    --output_file "$SLURM_JOBID" \
    --predict_position 0 \
    --sentences_on_batch "$sentences_on_batch" \
    --ner_model_dir "$ner_model" \

seff $SLURM_JOBID
gpuseff $SLURM_JOBID
echo "END $SLURM_JOBID: $(date)"
