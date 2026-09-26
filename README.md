# X Space English captions

Read live English captions from a Japanese X Space in your terminal. The tool uses `yt-dlp`, `ffmpeg`, and the OpenAI Realtime Translation API. You need a live Space, an X browser session, and an OpenAI API key with credits.

## Set up

Install `ffmpeg` and `yt-dlp`. On macOS, use:

```bash
brew install ffmpeg yt-dlp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key'
```

## Read captions

```bash
python relay.py 'https://x.com/i/spaces/SPACE_ID'
```

The tool uses Chrome cookies by default. To use a different browser, add `--cookies-from-browser firefox`. English text appears in the terminal. Press `Ctrl-C` to stop. Do not put your API key in Git.
