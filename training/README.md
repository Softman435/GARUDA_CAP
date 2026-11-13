# Fine-tune a speech model (and use it in GARUDA CAPTIONS AI)

This kit does not include training audio or transcripts. It can fine-tune one Whisper language per run when you provide
licensed audio paired with exact transcripts. Kannada, English, and Telugu need their own labeled recordings for
reliable language-specific fine-tuning. Kanglish is Kannada recognition followed by transliteration, not a separate
speech model.

**Be realistic:** training ("fine-tuning") needs an NVIDIA GPU. Your Intel i5 laptop is not suitable (it would take days).
Use a **free Google Colab GPU** for the training, then bring the finished model back to your computer.
Before you train anything, first try the ready-made Kannada models (`kn-medium`, see `docs\KANNADA_ACCURACY.md`):
they were already trained on large Kannada data and may be good enough.

**Status of this kit:** `make_metadata.py` is tested. `finetune_whisper.py` and the conversion are a standard recipe
that I could NOT run (no GPU or internet where this kit was written). Read what it prints; if it stops with an error,
send me the error text.

## What data do you need?
* Audio clips of **up to 30 seconds** each, clear speech, with the **exact text that is said**.
* For mixed speech, write English words in English letters inside the Kannada sentence, exactly as spoken.
* More is better. A few hours can adapt a model to your voice and topics; general improvement needs tens of hours.
* You must have the right to use the data (check the licence of the repository / dataset).

Folder layout (either one):
```
my_dataset/clip001.wav   my_dataset/clip001.txt      <- same name, text file holds what is said
my_dataset/clip002.mp3   my_dataset/clip002.txt
```
or a table (CSV) with a path column and a text column. Then create `metadata.csv`:
```
python training\make_metadata.py --data my_dataset
python training\make_metadata.py --data my_dataset --table transcripts.csv --path-col audio --text-col sentence
```

## Steps in Google Colab
1. Go to https://colab.research.google.com, **New notebook**, menu **Runtime > Change runtime type > T4 GPU**.
2. Upload the folder `training` and your `my_dataset` (zip them, upload, and unzip), or mount Google Drive.
3. Run these cells one by one:
```
!pip install -r training/requirements-colab.txt
!python training/finetune_whisper.py --data my_dataset --base vasista22/whisper-kannada-small --output out --max-steps 2000
```
   * Small dataset: keep `--base vasista22/whisper-kannada-small` (already Kannada-tuned) and `--max-steps 1000-2000`.
   * Large dataset: `--base openai/whisper-small` and `--max-steps 4000` or more.
   * If it says "CUDA out of memory": add `--batch 8` (or 4).
   * It prints the word error rate BEFORE and AFTER training. If AFTER is not lower, the data or settings are the problem.
4. When it prints `DONE`, download the folder `out/ct2` (right-click, Download, or zip it first).

## Use the model in the app
1. Copy the downloaded folder to `data\models\my-kannada-ct2` inside your GARUDA project.
2. Open the file `.env` and add the line:   `CUSTOM_MODEL=data\models\my-kannada-ct2`
3. Restart (`scripts\start.bat`). In the Transcript tab, choose the model **custom**.
4. Compare with the other models on files you did NOT train on:
```
scripts\evaluate.bat --data "C:\my-test-files" --models kn-medium custom --language kn
```
Keep some recordings aside for this test (never test on the training clips: the score would be fake).
