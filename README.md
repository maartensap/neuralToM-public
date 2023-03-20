# NeuralToM-v2

Example commands:

Running SocialIQa probing:
```bash
python gpt3LMMCprobing.py --task siqa --debug_dev 200 --n_examples 20 --model_variant gpt-3.5-turbo-0301 --probing_type mc --group_accuracy_by reasoningDim promptDim promptQuestionFocusChar --examples_of_same_type promptDim
```

Running ToMi probing:
```bash
python gpt3LMMCprobing.py --task tomi --debug_dev 200 --n_examples 10 --model_variant gpt-3.5-turbo-0301 --probing_type mc --group_accuracy_by falseTrueBelief factVsMind --examples_of_same_type qOrder --stratify_examples_by answerMemOrReal
```

Change `--probing_type`:
- lm: LM-probing (in NeuralToM EMNLP 2022 paper)
- mc: multiple-choice
- genli: generate+NLI (Xuhui, todo)
