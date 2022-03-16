# ResLM
Querying a Language model in a residual path for drug-target interaction task


### Codabase Descriptions:
* **Scripts folder**: Where you will find model skeleton used, training script, testing script, config files of hyperparameters.
* **Assets folder**: Where you will find the results of trainig(train_state ,learning_curve) and testing(evaluation_metrics).
* **Repo Commits**: Describe each experiments we conducted and their related config files and evaluation metrics.

### Compute:
* TPU v3-8 (Provided by TPU Research Cloud Program)

### Dataset:
* KIBA dataset: https://drive.google.com/drive/folders/1H92GKlu0Z6WFu9__ui1Ecwmq5nCMMk-Q?usp=sharing 

### Evaluation Metrics:
* **Mean Squared Error(MSE)**: It's the typical loss function.
* **Concordance Index(CI)**: Gives us the probability of the predicted KIBA interaction scores of two randomly chosen drug-target pairs, are in the correct order. This means if the reference KIBA score of one is greater than the other, this order should be preserved in the predicted scores of the two.
* **Area Under the Precision-Recall Curve(AUPR)**: Binary classification metric, here we transform the regression interaction KIBA scores to binary labels using
threshold values in related work. This threshold is 12.1 KIBA score, any drug-target pair with KIBA score greater or equal to this threshold are marked as there is binding between the pair, other than that as no-binding.

### ResLM Architecture:


![alt text](https://github.com/mhmdsabry/ResLM/blob/main/ResLM_Architecture/ResLM_RNN.drawio.png?raw=true)


**Note:** Frozen LM in ResLM architecture above means, we freeze all model parameters except for layernorm and positional encoding. Also we have tired all combinations of bias, layernorm and positional encoding, but we didn't see a noticable variation in performance. This fine-tuning techniques is inspired by work of https://arxiv.org/abs/2103.05247, https://arxiv.org/abs/2106.10199

