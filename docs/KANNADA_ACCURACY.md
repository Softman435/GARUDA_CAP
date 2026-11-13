# Getting better Kannada captions

English works well because Whisper was trained on a lot of English. Its general models saw very little Kannada, so
`tiny` and `base` are weak for Kannada, and even `large-v3` is far from perfect. These steps help, in this order.

## 1. Use a Kannada-tuned model (no training, try this first)
In the Transcript tab choose language **Kannada** and model **kn-medium**. For **Kannada + English (mixed)**, use
**large-v3** so the model can detect both languages instead of forcing Kannada.
* `kn-medium` is a Kannada fine-tune of Whisper medium made by SPRING Lab, IIT Madras (Apache-2.0), converted to the
  faster-whisper format by a third party (`itsskofficial/livewhisper-kn-medium` on Hugging Face).
* Its converter reports about 32% word error on Kannada test speech, against about 67% for the general large-v3 model.
  That is the converter's own measurement under their conditions, not mine. Measure on YOUR videos (step 3).
* First use downloads about 1.5 GB (needs internet once). It is stored in `data\models`.
* It is **slow on a CPU** (medium size). Test a 1-minute clip first; a long video can take longer than its own length.
* It was trained on Kannada, so Kannada is forced. English words inside a sentence may come out in Kannada letters.
  For mixed Kannada-English or mostly-English videos use the general `large-v3` model instead.

## 2. A faster Kannada model: kn-small (one-time conversion)
The smaller Kannada fine-tune only exists in Hugging Face format. Convert it once (internet needed, about 1 GB for a
temporary PyTorch environment):
```
scripts\convert_model.bat vasista22/whisper-kannada-small kn-small-ct2 small
```
Restart the app; **kn-small** then appears in the model list.

## 3. Measure instead of guessing
Put some recordings and the exact text of each in a folder (`clip.wav` + `clip.txt`, or a `metadata.csv`):
```
scripts\evaluate.bat --data "C:\my-test-files" --models base small kn-medium --language kn --limit 20
```
It prints the word error rate (WER) and character error rate (CER) of each model and saves a per-file CSV in
`data\evaluation`. Lower is better. WER is harsh for Kannada because words are long and joined, so watch CER too.

## 4. Teach it your words
Create `data\vocabulary.txt`, one name or special term per line (Kannada or English), for example people, places,
product names. Whisper is shown them before decoding, which helps with rare words. Keep the list short.

## 5. Train on your own repository / data
See `training\README.md` (free Google Colab GPU). Then set `CUSTOM_MODEL=` in `.env` and choose **custom**.

## 6. Always proof-read
No model is perfect for Kannada. The editor exists so you can correct the text; export only after checking.
