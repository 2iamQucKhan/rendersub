import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


def download_video(url, output_dir):
    """
    Tai video tu URL bang yt-dlp va tra ve duong dan file da tai.
    """
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Link video khong hop le. Vui long dung URL bat dau bang http:// hoac https://.")

    out_dir = Path(output_dir).expanduser() if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
        },
        "nocheckcertificate": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = Path(ydl.prepare_filename(info))
        filename_mp4 = filename.with_suffix(".mp4")

        if filename_mp4.exists():
            return str(filename_mp4)
        if filename.exists():
            return str(filename)

        for candidate in out_dir.iterdir():
            if candidate.is_file() and candidate.stem.startswith(filename.stem):
                return str(candidate)

    raise FileNotFoundError("Khong the tim thay file video da tai.")


def extract_audio(video_path, audio_path=None):
    """
    Trich xuat audio tu video sang WAV 16kHz mono, toi uu cho Whisper.
    """
    video = Path(video_path).expanduser()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"Khong tim thay file video: {video}")

    audio = Path(audio_path).expanduser() if audio_path else video.with_suffix(".wav")
    audio.parent.mkdir(parents=True, exist_ok=True)

    from resource_utils import get_ffmpeg_path
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("Khong tim thay ffmpeg. Vui long cai ffmpeg va them vao PATH.") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg khong the tach am thanh:\n{err[-1200:]}") from exc

    return str(audio)
