#!/usr/bin/env python3
"""Print English captions from a live Japanese X Space."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence

import websockets

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
CHUNK_BYTES = SAMPLE_RATE * SAMPLE_WIDTH // 5  # 200 ms of mono PCM16
TRANSLATION_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"


class RelayError(Exception):
    """A source, session, or process failure."""


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RelayError(f"{name} is not installed or is not on PATH")
    return path


def resolve_hls_url(space_url: str, browser: str | None) -> str:
    command = [executable("yt-dlp"), "--get-url", "--format", "bestaudio", "--no-playlist"]
    if browser:
        command.extend(("--cookies-from-browser", browser))
    result = subprocess.run([*command, space_url], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RelayError(result.stderr.strip() or "yt-dlp did not find live audio")
    urls = result.stdout.splitlines()
    if not urls:
        raise RelayError("yt-dlp did not find live audio")
    return urls[0]


def source_command(hls_url: str) -> list[str]:
    return [
        executable("ffmpeg"), "-hide_banner", "-loglevel", "error",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", hls_url, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]


class Captions:
    def __init__(self) -> None:
        self.characters = 0
        self.at_line_start = True

    def write(self, delta: str) -> None:
        if self.at_line_start:
            delta = delta.lstrip()
        if not delta:
            return
        # Preserve every fragment; only insert line breaks after sentence punctuation.
        text = re.sub(r"([.!?])(?=\s|$)", r"\1\n", delta)
        text = re.sub(r"\n\s+", "\n", text)
        sys.stdout.write(text)
        sys.stdout.flush()
        self.characters += len(delta)
        self.at_line_start = text.endswith("\n")

    def finish(self) -> None:
        if self.characters and not self.at_line_start:
            print(flush=True)
            self.at_line_start = True


async def expect_session_event(ws: websockets.ClientConnection, expected: str) -> None:
    try:
        event = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
    except asyncio.TimeoutError as error:
        raise RelayError(f"Timed out waiting for {expected}") from error
    except websockets.exceptions.ConnectionClosed as error:
        raise RelayError(f"OpenAI session closed: {error}") from error
    if event.get("type") == "error":
        detail = event.get("error", {})
        raise RelayError(f"OpenAI session rejected: {detail.get('code') or detail.get('type')}: {detail.get('message')}")
    if event.get("type") != expected:
        raise RelayError(f"Expected {expected}; received {event.get('type')}")


async def send_audio(ws: websockets.ClientConnection, source: asyncio.StreamReader, process: asyncio.subprocess.Process) -> int:
    while chunk := await source.read(CHUNK_BYTES):
        await ws.send(json.dumps({
            "type": "session.input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }))
    await ws.send(json.dumps({"type": "session.close"}))
    return await process.wait()


async def receive_captions(ws: websockets.ClientConnection, captions: Captions) -> None:
    async for message in ws:
        event = json.loads(message)
        kind = event.get("type")
        if kind == "session.output_transcript.delta":
            captions.write(event["delta"])
        elif kind == "error":
            detail = event.get("error", {})
            raise RelayError(f"OpenAI translation error: {detail.get('code') or detail.get('type')}: {detail.get('message')}")
        elif kind == "session.closed":
            return
    raise RelayError("OpenAI session ended without session.closed")


async def run(hls_url: str) -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RelayError("OPENAI_API_KEY is not set")
    captions = Captions()
    headers = {"Authorization": f"Bearer {key}", "OpenAI-Safety-Identifier": "x-space-translation-relay"}
    try:
        async with websockets.connect(TRANSLATION_URL, additional_headers=headers) as ws:
            await expect_session_event(ws, "session.created")
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"audio": {"output": {"language": "en"}}},
            }))
            await expect_session_event(ws, "session.updated")
            print("Connected. English captions appear below. Press Ctrl-C to stop.\n", file=sys.stderr, flush=True)
            process = await asyncio.create_subprocess_exec(*source_command(hls_url), stdout=asyncio.subprocess.PIPE)
            assert process.stdout is not None
            sender = asyncio.create_task(send_audio(ws, process.stdout, process))
            receiver = asyncio.create_task(receive_captions(ws, captions))
            try:
                done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
                if sender in done:
                    status = sender.result()
                    await receiver
                    if status:
                        raise RelayError(f"ffmpeg stopped with exit code {status}")
                else:
                    receiver.result()
                    if not sender.done():
                        raise RelayError("OpenAI session closed while source audio was active")
                    if sender.result():
                        raise RelayError(f"ffmpeg stopped with exit code {sender.result()}")
                if not captions.characters:
                    raise RelayError("No English captions arrived before the source ended")
            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                if process.returncode is None:
                    process.terminate()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                await process.wait()
    finally:
        captions.finish()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space_url", help="live X Space URL")
    parser.add_argument("--cookies-from-browser", default="chrome", metavar="BROWSER",
                        help="browser with your X session (default: chrome)")
    args = parser.parse_args(argv)
    hls_url = resolve_hls_url(args.space_url, args.cookies_from_browser)
    asyncio.run(run(hls_url))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RelayError, OSError, ValueError, websockets.WebSocketException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
