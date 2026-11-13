# Using your IndicConformer (.nemo) file

**What the file is:** AI4Bharat's *IndicConformer* hybrid CTC/RNNT speech model (MIT licence), about 120 million parameters,
**already trained** on Indian languages (the file lists 22; this app offers Kannada and 12 other languages). It needs **no
training** to be used. "Training" would only mean fine-tuning it further on your own recordings: that is a separate,
GPU-only job and is not part of this setup.

**Why it is set up this way:** the model only runs with AI4Bharat's own copy of NVIDIA NeMo (branch `nemo-v2`), which is built
for Linux. Plain Windows cannot install it. So the app uses **WSL2** (a Linux system inside Windows 11): the app cuts the
audio at pauses into chunks of at most 25 seconds, and a small worker inside WSL runs the model on each chunk.

**Honest status:** the bridge (chunking, worker protocol, progress, cancel, time limit, error messages) is tested with a
stand-in for NeMo. The real installation inside WSL and the real recognition accuracy have **not** been run by me: no
PyTorch, NeMo, internet or Windows was available to me. `scripts\diagnose.bat --indic` tells you whether it works on your PC.

## Steps (Windows 11)
1. **Put the model file in place.** Copy the `.nemo` file to `data\models\indic\` inside the project:
   ```
   New-Item -ItemType Directory -Force data\models\indic
   Copy-Item "C:\path\to\indicconformer_stt_kn_hybrid_rnnt_large.nemo" data\models\indic\
   ```
2. **Install WSL with Ubuntu 22.04** (once). Open PowerShell **as Administrator** and run:
   ```
   wsl --install -d Ubuntu-22.04
   ```
   Restart if asked. Open "Ubuntu 22.04" from the Start menu once and choose a username and password.
3. **Install the engine** (20-60 minutes, several GB, needs internet). In the project folder:
   ```
   .\scripts\setup_indic.bat
   ```
4. **Check it.** This must end with `READY`:
   ```
   .\scripts\diagnose.bat --indic
   ```
5. **Use it.** `.\scripts\start.bat`, then in the Transcript tab choose the language **Kannada** and the model **indic**.
6. **Compare with Whisper on your own recordings** (a folder with `clip.wav` + `clip.txt` pairs):
   ```
   .\scripts\evaluate.bat --data "C:\your-test-folder" --models kn-medium indic --language kn --limit 20
   ```

## What to expect
* Much faster than Whisper medium on a CPU, because it is not a word-by-word text generator.
* Captions are **phrase-level** (real start/end of each spoken chunk) unless the installed NeMo returns word times; then
  they are word-level. The app never invents word times.
* It writes Kannada in Kannada letters. English words spoken inside a sentence are written in Kannada letters or may be
  missed. For mostly-English videos choose a Whisper model.
* The model reads the audio in chunks cut at pauses. Videos with no pauses are cut into equal parts, which can split a word.
* Always proof-read: no speech model is perfect.

## Problems
| Message | What to do |
|---|---|
| `indic` is not in the model list | The `.nemo` file is not in `data\models\indic`. Copy it there and restart the app. |
| `IndicConformer is not set up yet` | Run `scripts\setup_indic.bat`. |
| `This is stock NVIDIA NeMo` | NeMo installed without AI4Bharat's changes. Run `scripts\setup_indic.bat` again. |
| `wsl` not found / Ubuntu not installed | Step 2 above (PowerShell as Administrator). |
| Setup stops at `reinstall.sh` | Send me the 20 lines around the first `ERROR`. Use Ubuntu **22.04** (Python 3.10). |
| Very slow first run | WSL reads the model through the Windows disk; the model loads once per transcription. |
| Out of memory | Close other programs; WSL uses up to half of your RAM by default. |
| Full details | `data\logs\garuda.log` |
