import os
import sys
import shutil
import subprocess
import time

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

def build():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    print("==================================================")
    print(" BẮT ĐẦU QUY TRÌNH BUILD STANDALONE RENDERSUB (WIN) ")
    print("==================================================")

    # 1. Clean build & dist
    for d in ["build", "dist"]:
        dp = os.path.join(root_dir, d)
        if os.path.exists(dp):
            print(f"🧹 Đang dọn dẹp thư mục {d}/...")
            shutil.rmtree(dp, ignore_errors=True)

    # 2. Verify FFmpeg binaries exist
    ffmpeg_src = os.path.join(root_dir, "bin", "ffmpeg.exe")
    ffprobe_src = os.path.join(root_dir, "bin", "ffprobe.exe")
    if not (os.path.exists(ffmpeg_src) and os.path.exists(ffprobe_src)):
        print("⚠️ Cảnh báo: bin/ffmpeg.exe hoặc bin/ffprobe.exe chưa có sẵn trong thư mục bin/")

    # 3. Invoke PyInstaller
    print("📦 Đang biên dịch với PyInstaller (rendersub.spec)...")
    t0 = time.time()
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "rendersub.spec"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("❌ Lỗi: Quá trình PyInstaller thất bại!")
        sys.exit(1)

    build_time = time.time() - t0
    print(f"✔ Hoàn thành biên dịch PyInstaller trong {build_time:.1f}s")

    # 4. Ensure bin/, Data/, and config/ are in dist/RenderSub/
    dist_rendersub = os.path.join(root_dir, "dist", "RenderSub")
    dist_bin = os.path.join(dist_rendersub, "bin")
    os.makedirs(dist_bin, exist_ok=True)
    if os.path.exists(ffmpeg_src) and not os.path.exists(os.path.join(dist_bin, "ffmpeg.exe")):
        shutil.copy2(ffmpeg_src, os.path.join(dist_bin, "ffmpeg.exe"))
    if os.path.exists(ffprobe_src) and not os.path.exists(os.path.join(dist_bin, "ffprobe.exe")):
        shutil.copy2(ffprobe_src, os.path.join(dist_bin, "ffprobe.exe"))

    if os.path.exists(os.path.join(root_dir, "Data")):
        shutil.copytree(os.path.join(root_dir, "Data"), os.path.join(dist_rendersub, "Data"), dirs_exist_ok=True)
    if os.path.exists(os.path.join(root_dir, "config")):
        shutil.copytree(os.path.join(root_dir, "config"), os.path.join(dist_rendersub, "config"), dirs_exist_ok=True)

    # 5. Verify standalone output
    exe_path = os.path.join(dist_rendersub, "RenderSub.exe")
    if not os.path.exists(exe_path):
        print("❌ Không tìm thấy RenderSub.exe trong dist/RenderSub/!")
        sys.exit(1)

    print("==================================================")
    print(" BUILD THÀNH CÔNG: dist/RenderSub/RenderSub.exe")
    print("==================================================")

if __name__ == "__main__":
    build()
