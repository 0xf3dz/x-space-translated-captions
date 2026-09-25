#!/usr/bin/env python3
"""Relay a live HLS audio stream through OpenAI realtime translation and laptop speakers."""

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
CHUNK_MS = 200
CHUNK_BYTES = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * CHUNK_MS // 1_000
DEFAULT_SPEAKER_VOLUME = 60
TRANSLATION_URL = (
    "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
)


class RelayError(Exception):
    """A configuration or child-process error."""

class RelayStatus:
    def __init__(self) -> None:
        self.input_bytes = 0
        self.output_bytes = 0
        self.transcript_fragments = 0


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def report_progress(status: RelayStatus) -> None:
    while True:
        await asyncio.sleep(15)
        report(
            f"Relay: {status.input_bytes / (SAMPLE_RATE * SAMPLE_WIDTH):.0f}s input, "
            f"{status.output_bytes} English audio bytes, "
            f"{status.transcript_fragments} transcript fragments"
        )


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


def speaker_command(volume: int) -> list[str]:
    return [
        executable("ffplay"),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nodisp",
        "-volume",
        str(volume),
        "-f",
        "s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ch_layout",
        "mono",
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


async def stream_audio(
    websocket: websockets.ClientConnection,
    source: asyncio.StreamReader,
    source_process: asyncio.subprocess.Process,
    status: RelayStatus,
) -> int:
    while chunk := await source.read(CHUNK_BYTES):
        await websocket.send(
            json.dumps(
                {
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        status.input_bytes += len(chunk)
        if status.input_bytes == len(chunk):
            report("Source audio reached the translation session.")
    await websocket.send(json.dumps({"type": "session.close"}))
    return await source_process.wait()


async def receive_events(
    websocket: websockets.ClientConnection,
    speaker: asyncio.subprocess.Process,
    transcript: TranscriptLog,
    status: RelayStatus,
) -> None:
    assert speaker.stdin is not None
    async for message in websocket:
        event = json.loads(message)
        event_type = event.get("type")
        if event_type == "session.updated":
            report("English translation session configured.")
        elif event_type == "session.output_audio.delta":
            audio = base64.b64decode(event["delta"])
            try:
                speaker.stdin.write(audio)
                await speaker.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as error:
                state = "unknown status" if speaker.returncode is None else f"exit code {speaker.returncode}"
                raise RelayError(f"ffplay stopped while playing translated audio ({state})") from error
            status.output_bytes += len(audio)
            if status.output_bytes == len(audio):
                report("English audio reached the laptop output.")
        elif event_type in {
            "session.input_transcript.delta",
            "session.output_transcript.delta",
        }:
            transcript.write(event_type, event["delta"])
            status.transcript_fragments += 1
        elif event_type == "error":
            raise RelayError(json.dumps(event, ensure_ascii=False))
        elif event_type == "session.closed":
            return
    raise RelayError("Translation session ended without session.closed")


async def run_relay(hls_url: str, transcript_path: Path | None, speaker_volume: int) -> None:
    api_key = require_api_key()
    source = await asyncio.create_subprocess_exec(
        *source_command(hls_url),
        stdout=asyncio.subprocess.PIPE,
    )
    speaker = await asyncio.create_subprocess_exec(
        *speaker_command(speaker_volume),
        stdin=asyncio.subprocess.PIPE,
    )
    assert source.stdout is not None
    transcript = TranscriptLog(transcript_path)
    status = RelayStatus()
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
                        "session": {
                            "audio": {
                                "input": {"transcription": {"model": "gpt-realtime-whisper"}},
                                "output": {"language": "en"},
                            }
                        },
                    }
                )
            )
            report("Connected to OpenAI; waiting for source audio.")
            send_task = asyncio.create_task(stream_audio(websocket, source.stdout, source, status))
            receive_task = asyncio.create_task(receive_events(websocket, speaker, transcript, status))
            progress_task = asyncio.create_task(report_progress(status))
            try:
                done, _ = await asyncio.wait(
                    (send_task, receive_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if send_task in done:
                    source_status = send_task.result()
                    await receive_task
                    if source_status != 0:
                        raise RelayError(f"ffmpeg stopped with exit code {source_status}")
                else:
                    receive_task.result()
            finally:
                progress_task.cancel()
                for task in (send_task, receive_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(send_task, receive_task, progress_task, return_exceptions=True)
    finally:
        transcript.close()
        if speaker.stdin:
            speaker.stdin.close()
        if source.returncode is None:
            source.terminate()
        if speaker.returncode is None:
            speaker.terminate()
        await source.wait()
        await speaker.wait()


def validate_volume(value: int) -> int:
    if not 0 <= value <= 100:
        raise RelayError("--volume must be between 0 and 100")
    return value


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
    print("Set the laptop output to its built-in speakers, not Bluetooth earbuds.")
    print("Place the host phone near the laptop speaker and mute host-phone playback.")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


async def play_tone(seconds: int, volume: int) -> int:
    process = await asyncio.create_subprocess_exec(
        executable("ffplay"),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nodisp",
        "-autoexit",
        "-volume",
        str(volume),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:sample_rate={SAMPLE_RATE}:duration={seconds}",
    )
    return await process.wait()


def test_tone(seconds: int, volume: int) -> int:
    return asyncio.run(play_tone(seconds, volume))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="translate a live Space into laptop speaker audio")
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
    run.add_argument(
        "--volume",
        type=int,
        default=DEFAULT_SPEAKER_VOLUME,
        metavar="0-100",
        help="laptop speaker volume for translated audio (default: %(default)s)",
    )

    commands.add_parser("doctor", help="check required local software and configuration")
    tone = commands.add_parser("test-tone", help="play a 1 kHz tone through the laptop speakers")
    tone.add_argument("--seconds", type=int, default=5)
    tone.add_argument(
        "--volume",
        type=int,
        default=DEFAULT_SPEAKER_VOLUME,
        metavar="0-100",
        help="laptop speaker volume (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "doctor":
        return doctor()
    if args.command == "test-tone":
        if args.seconds < 1:
            raise RelayError("--seconds must be at least 1")
        return test_tone(args.seconds, validate_volume(args.volume))
    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
    hls_url = args.hls_url or resolve_hls_url(args.space_url, args.cookies_from_browser)
    asyncio.run(run_relay(hls_url, args.transcript, validate_volume(args.volume)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RelayError, OSError, ValueError, websockets.WebSocketException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
