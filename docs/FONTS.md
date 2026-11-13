# Fonts and Indian languages

Captions are drawn with the fonts in `backend/fonts/` (`.ttf` / `.otf`). Missing glyphs show as empty boxes, so the
app checks your text against the chosen font and warns you in the editor when it cannot draw a character.

## What is included
- **DejaVu Sans** (Regular + Bold): Latin/English only. Included so exports always work for English.
- `scripts\install.bat` copies **Nirmala UI** from your own Windows folder (`C:\Windows\Fonts`), including the `.ttc` font collection used by current Windows releases. The app reads `.ttc`, `.ttf` and `.otf` fonts. It covers Devanagari (Hindi, Marathi, Sanskrit),
  Bengali/Assamese, Gujarati, Gurmukhi (Punjabi), Kannada, Malayalam, Odia, Tamil and Telugu. It is Microsoft's font, copied for your own use and **not** redistributed in this zip.
  In the Style tab choose **Nirmala UI** for Indian-language captions.

## Better looking (free, open source) fonts
Download from https://fonts.google.com the **Noto Sans Kannada**, **Noto Sans Devanagari**, etc. you need
(licence: SIL Open Font License). Put the **static** `.ttf` files (not the variable ones if you can avoid them)
into `backend/fonts/` and restart the app. They appear in the Font list automatically.

## Always test
Before exporting a long video: transcribe or type a few Kannada/Hindi captions, look at the preview, export a 10-second test and watch it.
Nothing in this project has verified Kannada/Hindi rendering yet. That check is yours to do on first use.
Complex scripts (conjuncts, vowel signs) are rendered by FFmpeg's libass with HarfBuzz; if you see broken shaping, try another font and report which build of FFmpeg you use.
