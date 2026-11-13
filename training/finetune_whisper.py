"""Fine-tune a Whisper model on your own Kannada (or Kannada + English) audio and convert it for GARUDA CAPTIONS AI.

RUN THIS ON A GPU (a free Google Colab T4 works). It is far too slow for a laptop CPU.
It has NOT been run by the author of this kit (no GPU or internet was available): treat it as a standard, readable
starting point and read the messages it prints. See training/README.md for the exact Colab steps.

  python finetune_whisper.py --data /content/my_dataset --base vasista22/whisper-kannada-small --output /content/out

--data   a folder with metadata.csv (columns file_name,text) next to the audio, or a Hugging Face dataset id
--base   the model to start from. With a small dataset, start from an already Kannada-tuned model
         (vasista22/whisper-kannada-small). Otherwise use openai/whisper-small.
"""
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, DatasetDict, load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer)


@dataclass
class Collator:
    """Pads audio features and label ids (labels padded with -100 so the padding is ignored in the loss)."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad([{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--language", default="Kannada", help="language NAME as Whisper spells it (e.g. Kannada)")
    ap.add_argument("--output", default="whisper-kannada-finetuned")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16, help="lower this (8, 4) if the GPU runs out of memory")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--test-size", type=float, default=0.1, help="share kept aside to measure accuracy (if no 'test' split exists)")
    ap.add_argument("--no-convert", action="store_true", help="skip the CTranslate2 conversion at the end")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no GPU found. Training on a CPU would take days. In Colab: Runtime > Change runtime type > GPU.")

    # ---- data ------------------------------------------------------------------------------------------------
    path = Path(args.data)
    data = load_dataset("audiofolder", data_dir=str(path)) if path.is_dir() else load_dataset(args.data)
    if "test" not in data and "validation" not in data:
        data = data["train"].train_test_split(test_size=args.test_size, seed=42)
    eval_key = "test" if "test" in data else "validation"
    data = data.cast_column("audio", Audio(sampling_rate=16000))
    print({k: len(v) for k, v in data.items()})

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base)
    tokenizer = WhisperTokenizer.from_pretrained(args.base, language=args.language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.base, language=args.language, task="transcribe")

    def prepare(batch: dict) -> dict:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch[args.text_column]).input_ids
        return batch

    data = data.map(prepare, remove_columns=data[eval_key].column_names, num_proc=1)

    # ---- model -----------------------------------------------------------------------------------------------
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = args.language.lower()
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    import evaluate
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict:
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=predictions, references=references)}

    kwargs = dict(
        output_dir=args.output, per_device_train_batch_size=args.batch, gradient_accumulation_steps=max(1, 16 // args.batch),
        learning_rate=args.lr, warmup_steps=min(500, args.max_steps // 10), max_steps=args.max_steps,
        gradient_checkpointing=True, fp16=torch.cuda.is_available(), per_device_eval_batch_size=8,
        predict_with_generate=True, generation_max_length=225, save_steps=max(200, args.max_steps // 4),
        eval_steps=max(200, args.max_steps // 4), logging_steps=25, report_to=[], load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False, save_total_limit=2, remove_unused_columns=False,
    )
    arg_names = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in arg_names else "evaluation_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"
    trainer_args = dict(args=Seq2SeqTrainingArguments(**kwargs), model=model, train_dataset=data["train"],
                        eval_dataset=data[eval_key], data_collator=Collator(processor, model.config.decoder_start_token_id),
                        compute_metrics=compute_metrics)
    # transformers renamed `tokenizer` to `processing_class`
    trainer_args["processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters else "tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_args)

    print("Accuracy BEFORE training (word error rate, lower is better):", trainer.evaluate()["eval_wer"])
    trainer.train()
    print("Accuracy AFTER training:", trainer.evaluate()["eval_wer"])

    final = Path(args.output) / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))          # writes tokenizer.json and preprocessor_config.json
    print("Saved the fine-tuned model to", final)

    # ---- convert for GARUDA CAPTIONS AI ----------------------------------------------------------------------
    if not args.no_convert:
        from ctranslate2.converters import TransformersConverter
        out = Path(args.output) / "ct2"
        TransformersConverter(str(final), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(str(out), quantization="int8", force=True)
        print("\nDONE. Download this folder and put it in the app's data\\models folder:", out)


if __name__ == "__main__":
    main()
