# From Rules to Reasoning: Empowering LLMs with Contrastive Learning for Effective Hate Speech Moderation

This repository contains the code for the paper "From Rules to Reasoning: Empowering LLMs with Contrastive Learning for Effective Hate Speech Moderation".

## Paper - 

## Code
To run the code, you need to have Python 3.8 or later installed. You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

### Preprocessing datasets
``` bash
python data_preprocessing.py -dp <dataset_path> -op <output_path>
```

### Model Inference
```bash 
python model_inference.py -dp <dataset_path> -mn <model_name> -pp <prompt_path> -op <output_path>
```

### Model Inference with Retrieval
```bash
python model_inference_w_retrieval.py -train <train_dataset_path> -test <test_dataset_path> -mn <model_name> -pp <prompt_path> -op <output_path>
```

### Model Evaluation
```bash
python evaluation.py -fp <output_file_path>
```
