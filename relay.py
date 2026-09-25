#!/usr/bin/env python3
"""Relay a live HLS audio stream through OpenAI realtime translation."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import websockets

SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_MS = 100
CHUNK_BYTES = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * CHUNK_MS // 1_000
TRANSLATION_URL = (
    "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
)


class RelayError(Exception):
    """A configuration or child-process error."""


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RelayError(f"{name} is not installed or is not on PATH")
    return path


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RelayError("OPENAI_API_KEY is not set")
    return api_key


def resolve_hls_url(space_url: str, cookie_browser: str | None) -> str:
    command = [
        executable("yt-dlp"),
        "--get-url",
        "--format",
        "bestaudio",
        "--no-playlist",
    ]
    if cookie_browser:
        command.extend(("--cookies-from-browser", cookie_browser))
    command.append(space_url)
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode:
        detail = result.stderr.strip() or "yt-dlp did not return a media URL"
        raise RelayError(detail)
    urls = result.stdout.splitlines()
    if not urls:
        raise RelayError("yt-dlp did not return a media URL")
    return urls[0]


def source_command(hls_url: str) -> list[str]:
    return [
        executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
        "-i",
        hls_url,
        "-vn",
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]


def player_command() -> list[str]:
    return [
        executable("ffplay"),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nodisp",
        "-f",
        "s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(CHANNELS),
        "-i",
        "pipe:0",
    ]


class TranscriptLog:
    def __init__(self, path: Path | None) -> None:
        self._file = path.open("a", encoding="utf-8") if path else None

    def write(self, event_type: str, delta: str) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps({"type": event_type, "delta": delta}, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


async def stream_audio(websocket: websockets.ClientConnection, source: asyncio.StreamReader) -> None:
    while chunk := await source.read(CHUNK_BYTES):
        await websocket.send(
            json.dumps(
                {
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
    await websocket.send(json.dumps({"type": "session.close"}))


async def receive_events(
    websocket: websockets.ClientConnection,
    player: asyncio.subprocess.Process,
    transcript: TranscriptLog,
) -> None:
    assert player.stdin is not None
    async for message in websocket:
        event = json.loads(message)
        event_type = event.get("type")
        if event_type == "session.output_audio.delta":
            player.stdin.write(base64.b64decode(event["delta"]))
            await player.stdin.drain()
        elif event_type in {
            "session.input_transcript.delta",
            "session.output_transcript.delta",
        }:
            transcript.write(event_type, event["delta"])
        elif event_type == "error":
            raise RelayError(json.dumps(event, ensure_ascii=False))
        elif event_type == "session.closed":
            return


async def run_relay(hls_url: str, transcript_path: Path | None) -> None:
    api_key = require_api_key()
    source = await asyncio.create_subprocess_exec(
        *source_command(hls_url),
        stdout=asyncio.subprocess.PIPE,
    )
    player = await asyncio.create_subprocess_exec(
        *player_command(),
        stdin=asyncio.subprocess.PIPE,
    )
    assert source.stdout is not None
    transcript = TranscriptLog(transcript_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Safety-Identifier": "x-space-translation-relay",
    }
    try:
        async with websockets.connect(TRANSLATION_URL, additional_headers=headers) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"audio": {"output": {"language": "en"}}},
                    }
                )
            )
            await asyncio.gather(
                stream_audio(websocket, source.stdout),
                receive_events(websocket, player, transcript),
            )
    finally:
        transcript.close()
        if player.stdin:
            player.stdin.close()
        if source.returncode is None:
            source.terminate()
        if player.returncode is None:
            player.terminate()
        await source.wait()
        await player.wait()


def doctor() -> int:
    failures: list[str] = []
    for tool in ("ffmpeg", "ffplay"):
        try:
            print(f"OK: {tool}: {executable(tool)}")
        except RelayError as error:
            failures.append(str(error))
    if os.environ.get("OPENAI_API_KEY"):
        print("OK: OPENAI_API_KEY is set")
    else:
        failures.append("OPENAI_API_KEY is not set")
    try:
        print(f"OPTIONAL: yt-dlp: {executable('yt-dlp')}")
    except RelayError:
        print("OPTIONAL: yt-dlp is absent; use --hls-url instead of --space-url")
    if sys.platform == "darwin":
        print("Set macOS Sound Output to the laptop headphone output before you start the relay.")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


async def play_tone(seconds: int) -> int:
    process = await asyncio.create_subprocess_exec(
        executable("ffplay"),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nodisp",
        "-autoexit",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:sample_rate={SAMPLE_RATE}:duration={seconds}",
    )
    return await process.wait()


def test_tone(seconds: int) -> int:
    return asyncio.run(play_tone(seconds))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="translate a live Space into English")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--hls-url", help="live HLS URL from the source Space")
    source.add_argument("--space-url", help="live X Space URL; requires yt-dlp")
    run.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="browser name for yt-dlp cookies, such as chrome or firefox",
    )
    run.add_argument(
        "--transcript",
        type=Path,
        help="newline-delimited JSON transcript output",
    )

    commands.add_parser("doctor", help="check required local software and configuration")
    tone = commands.add_parser("test-tone", help="play a 1 kHz tone through the default laptop output")
    tone.add_argument("--seconds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "doctor":
        return doctor()
    if args.command == "test-tone":
        if args.seconds < 1:
            raise RelayError("--seconds must be at least 1")
        return test_tone(args.seconds)
    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
    hls_url = args.hls_url or resolve_hls_url(args.space_url, args.cookies_from_browser)
    asyncio.run(run_relay(hls_url, args.transcript))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RelayError, OSError, ValueError, websockets.WebSocketException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
