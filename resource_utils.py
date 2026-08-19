import os
import sys
import shutil

def get_base_dir() -> str:
    """
    Trả về thư mục gốc của ứng dụng:
    - Khi chạy qua PyInstaller (.exe / onedir): trả về thư mục chứa file .exe hoặc sys._MEIPASS
    - Khi chạy dev (Python source): trả về thư mục chứa mã nguồn project
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(relative_path: str) -> str:
    """
    Trả về đường dẫn tuyệt đối tới tài nguyên tĩnh (Data/, config/, assets/, fonts/):
    - Ưu tiên tìm trong sys._MEIPASS (khi đóng gói)
    - Nếu không có, tìm trong thư mục cạnh file exe hoặc source code
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        candidate = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(candidate):
            return candidate
            
    base = get_base_dir()
    candidate = os.path.join(base, relative_path)
    return candidate

def get_user_data_dir() -> str:
    """
    Trả về thư mục có quyền ghi cho người dùng để lưu app_settings.json, logs, cache:
    - Windows: %LOCALAPPDATA%/RenderSub
    - Fallback: ~/.rendersub
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        target_dir = os.path.join(local_app_data, "RenderSub")
    else:
        target_dir = os.path.join(os.path.expanduser("~"), ".rendersub")
        
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "output"), exist_ok=True)
    return target_dir

def get_ffmpeg_path() -> str:
    """
    Tìm binary ffmpeg.exe theo thứ tự ưu tiên:
    1. <application_dir>/bin/ffmpeg.exe
    2. <_MEIPASS>/bin/ffmpeg.exe
    3. Thư mục bin/ trong mã nguồn
    4. System PATH qua shutil.which("ffmpeg")
    5. Fallback đường dẫn chuẩn của Windows Winget/Gyan nếu có
    """
    candidates = []
    
    # 1. Bên cạnh EXE
    if getattr(sys, 'frozen', False):
        candidates.append(os.path.join(os.path.dirname(sys.executable), "bin", "ffmpeg.exe"))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"))
        
    # 2. Trong _MEIPASS bundle
    if hasattr(sys, '_MEIPASS'):
        candidates.append(os.path.join(sys._MEIPASS, "bin", "ffmpeg.exe"))
        candidates.append(os.path.join(sys._MEIPASS, "ffmpeg.exe"))
        
    # 3. Trong thư mục mã nguồn
    src_base = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(src_base, "bin", "ffmpeg.exe"))
    
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
            
    # 4. Tìm trong System PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
        
    # 5. Fallback tìm trong Winget Gyan nếu người dùng cài qua winget
    winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_base):
        for root, dirs, files in os.walk(winget_base):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")
                
    raise RuntimeError(
        "Không tìm thấy công cụ FFmpeg (ffmpeg.exe).\n"
        "Vui lòng đảm bảo tệp 'bin/ffmpeg.exe' đi kèm ứng dụng hoặc đã cài đặt FFmpeg trên máy tính."
    )

def get_ffprobe_path() -> str:
    """
    Tìm binary ffprobe.exe theo thứ tự ưu tiên tương tự ffmpeg.
    """
    candidates = []
    
    if getattr(sys, 'frozen', False):
        candidates.append(os.path.join(os.path.dirname(sys.executable), "bin", "ffprobe.exe"))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "ffprobe.exe"))
        
    if hasattr(sys, '_MEIPASS'):
        candidates.append(os.path.join(sys._MEIPASS, "bin", "ffprobe.exe"))
        candidates.append(os.path.join(sys._MEIPASS, "ffprobe.exe"))
        
    src_base = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(src_base, "bin", "ffprobe.exe"))
    
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
            
    found = shutil.which("ffprobe")
    if found:
        return found
        
    winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_base):
        for root, dirs, files in os.walk(winget_base):
            if "ffprobe.exe" in files:
                return os.path.join(root, "ffprobe.exe")
                
    raise RuntimeError(
        "Không tìm thấy công cụ FFprobe (ffprobe.exe).\n"
        "Vui lòng đảm bảo tệp 'bin/ffprobe.exe' đi kèm ứng dụng hoặc đã cài đặt FFmpeg trên máy tính."
    )

def setup_app_environment():
    """
    Khởi tạo môi trường khi app bắt đầu:
    - Thêm thư mục bin/ chứa ffmpeg vào PATH nếu có
    - Cấu hình thư mục EasyOCR model nếu cần
    """
    try:
        ffmpeg_exe = get_ffmpeg_path()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
