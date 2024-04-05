from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines.pt_utils import KeyDataset
import transformers
import json
import pandas as pd
from datasets import Dataset
import argparse

transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)

class ModelInference:
    def __init__(self, dataset_path, model_name, prompt_path, output_path) -> None:
        self.access_token = "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"
        self.data_filepath = dataset_path
        self.base_model_id = model_name
        self.prompt_filepath = prompt_path
        self.output_path = output_path

    def get_data(self):
        df = pd.read_json(self.data_filepath)
        return df

    def get_model_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, model_max_length=720, padding_side="left", add_eos_token=True, 
            # add_bos_token=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, device_map="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", token=self.access_token)
        return pipe

    def get_prompt(self):
        with open(self.prompt_filepath) as f:
            prompt = f.read()
        return prompt

    def generate_output(self, text_generation_pipeline, dataset):
        predictions = []
        texts = []
        labels = []
        for i, out in enumerate(
            text_generation_pipeline(
                KeyDataset(dataset, "abuse_prompt"),
                batch_size=8,
                max_new_tokens=10,
                temperature=0.01,
                top_k=50,
                top_p=0.95,
                return_full_text=False,
                do_sample=True,
            )
        ):
            response = out[0]["generated_text"]
            predictions.append(response)
            texts.append(dataset[i]["text"])
            labels.append(dataset[i]["label"])
        out_df = pd.DataFrame()
        out_df["text"] = texts
        out_df["label"] = labels
        out_df["prediction"] = predictions
        return out_df

    def predict_labels(self):
        df = self.get_data()
        prompt = self.get_prompt()
        df["abuse_prompt"] = df["text"].apply(lambda x: prompt.format(input_sentence=x))
        text_generation_pipeline = self.get_model_pipeline()
        dataset = Dataset.from_pandas(df)
        out_df = self.generate_output(text_generation_pipeline, dataset)
        return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp","--dataset_path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("-mn","--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("-pp","--prompt_path", type=str, required=True, help="Path to the prompt file")
    parser.add_argument("-op","--output_path", type=str, required=True, help="Path to save the output file")
    args = parser.parse_args()

    model_pipeline = ModelInference(args.dataset_path, args.model_name, args.prompt_path, args.output_path)
    out_df = model_pipeline.predict_labels()
    out_df.to_csv(args.output_path, index=False)