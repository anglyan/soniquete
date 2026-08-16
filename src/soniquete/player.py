"""A simple cross-platform wav playback"""

from __future__ import annotations

import platform
import shutil
import subprocess


def play_wav(path: str) -> None:
    """Play the WAV file at ``path`` using the operating system's audio output.

    Works out of the box on macOS (via ``afplay``); also supports Linux
    (via ``aplay``/``paplay``) and Windows (via the ``winsound`` module).
    """
    system = platform.system()

    if system == "Darwin":
        subprocess.run(["afplay", path], check=True)
    elif system == "Linux":
        player = shutil.which("aplay") or shutil.which("paplay")
        if player is None:
            raise RuntimeError(
                "No audio player found; install 'aplay' or 'paplay'."
            )
        subprocess.run([player, path], check=True)
    elif system == "Windows":
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        raise RuntimeError(f"Unsupported platform: {system!r}")
