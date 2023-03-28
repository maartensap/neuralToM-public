python gpt3LMMCprobing.py --task siqa \
    --debug_dev 50 \
    --n_examples 0 \
    --model_variant gpt-3.5-turbo-0301 \
    --probing_type nli \
    --group_accuracy_by reasoningDim promptDim promptQuestionFocusChar \
    --output_prediction_file predictions/siqa_gpt3.5-turbo-0301_nli.csv \
