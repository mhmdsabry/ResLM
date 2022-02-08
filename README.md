# ResLM
Querying a Language model in a residual path for drug-target interaction task


### Codabase Descriptions:
* **Scripts folder**: Where you will find model skeleton used, training script, testing script, config files of hyperparameters.
* **Assets folder**: Where you will find the results of trainig(train_state ,learning_curve) and testing(evaluation_metrics).


### ResLM Architecture:


![alt text](https://github.com/mhmdsabry/ResLM/blob/main/ResLM_Architecture/ResLM_RNN.drawio.png?raw=true)


**Note:** Frozen LM in ResLM architecture above means, we freeze all model parameters except for layernorm and positional encoding. Also we have tired all combinations of bias, layernorm and positional encoding, but we didn't see a noticable variation in performance. This fine-tuning techniques is inspired by work of https://arxiv.org/abs/2103.05247, https://arxiv.org/abs/2106.10199

