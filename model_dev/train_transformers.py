from transformers import (Trainer, AutoModelForSequenceClassification, TrainingArguments,
                          AutoTokenizer, DataCollatorWithPadding, pipeline)
import evaluate
import mlflow
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import torch
#from mlflow.transformers import 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_FILES = {
    "train": "dataset/train.csv",
    "test" : "dataset/test.csv"
}

EXPERIMENT_NAME = "tranformers_comments"
SEQUENCE_LENGTH = 40
BATCH_SIZE = 32

mlflow.set_experiment(EXPERIMENT_NAME)

data = load_dataset("csv", data_files=DATA_FILES)

model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=3)
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
data_coll = DataCollatorWithPadding(tokenizer=tokenizer)

acc = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print("called")
    return acc.compute(predictions=predictions, references=labels)


aligment = {
            "Positive": 0,
            "Neutral" : 1,
            "Negative": 2
        }

def tokenize_dataset(dataset):
    return tokenizer(dataset["CommentText"], truncation=True)

data = data.map(tokenize_dataset, batched=True)
data = data.rename_columns(
    {"CommentText": "text",
     "Sentiment"  : "labels"}
)
data = data.class_encode_column("labels")
data = data.align_labels_with_mapping(aligment, "labels")
data["test"] = data["test"].select([0,1,2,3])

with mlflow.start_run():
    run_name = mlflow.active_run().info.run_name
    mlflow.transformers.autolog(log_models=True,
                                log_model_signatures=True)

    tr_args = TrainingArguments(
        output_dir=os.path.join("model_dev",
                                "transformer_results",
                                f"{run_name}"),
        #learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        fp16=True
    )

    trainer = Trainer(
        model = model,
        args = tr_args,
        train_dataset = data["test"],
        eval_dataset  = data["test"],
        processing_class= tokenizer,
        data_collator = data_coll,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    model.config.id2label = {
             0 : "Positive",
             1 : "Neutral",
             2 : "Negative"
        }
    pip = pipeline(task = "sentiment-analysis",
                   model = model,
                   tokenizer = tokenizer
                   )

    output = mlflow.transformers.generate_signature_output(pip, "hello world")
    signature = mlflow.models.infer_signature("hello world",
                                              output)
    model_info = mlflow.transformers.log_model(pip,
                                 name=f"transformer-{run_name}",
                                 #input_example="Hello world!",
                                 signature=signature)
    
    mlflow.register_model(
        model_uri= model_info.model_uri,
        name = f"transformer-{run_name}"
    )
    