# Windows 11 setup (simple steps)

You need three programs. Install them once, then run `scripts\install.bat`.

## 1. Python 3.12 (3.10 - 3.13 work)
1. Go to https://www.python.org/downloads/windows/ and download Python **3.12**.
2. Run the installer. **Tick "Add python.exe to PATH"** on the first screen, then click Install.
3. Check: open **Command Prompt** (Windows key, type `cmd`, Enter) and type `python --version`.

If you already have Python 3.13, the installer uses it but prints a note: 3.12 is the better-tested choice.
Having several Python versions is fine; `install.bat` picks 3.12, then 3.11, 3.10, 3.13.

## 2. Node.js (LTS)
1. Go to https://nodejs.org and download the **LTS** version. Install with the defaults.
2. Check: new Command Prompt, type `node --version`. (Needed once, to build the web page.)

## 3. FFmpeg (includes FFprobe)
Windows 11 has `winget`. In Command Prompt run:

    winget install Gyan.FFmpeg

**Close the window and open a NEW one**, then check both commands print a version:

    ffmpeg -version
    ffprobe -version

If Windows says "not recognized", FFmpeg is not on PATH: restart the computer, or add its `bin` folder to PATH
(Start, type "Edit the system environment variables", Environment Variables, Path, New, paste the folder that contains `ffmpeg.exe`).

## 4. Install the app
Double-click `scripts\install.bat` (or run it in a terminal to see messages). It:
* finds Python and creates `backend\.venv` (re-uses it if it exists),
* installs the packages (5-10 minutes; faster-whisper is large),
* checks FFmpeg/FFprobe and tells you how to install them if missing,
* builds the web page, creates `.env`, copies the Windows Kannada/Hindi font (Nirmala UI).

## 5. Check everything
Double-click `scripts\diagnose.bat`. It prints your versions, settings and any problem. It downloads nothing.

## 6. Start
Double-click `scripts\start.bat`. The browser opens at http://127.0.0.1:8000. Keep the black window open.

## The speech model
It is **not** inside the ZIP. It downloads automatically the first time you press "Generate captions"
(internet needed once; tiny about 75 MB, base about 145 MB, small about 480 MB). It is saved in `data\models`
inside this folder. To test it without a video first: `scripts\diagnose.bat --deep`.
