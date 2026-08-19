import os
import json
import requests
from translator import TrendingSlangManager

# Bộ từ điển Trend Slang Gen Z (Bilibili, Douyin, Xiaohongshu) 50+ từ cực hot
EXPANDED_TRENDING_SLANGS = {
    "破防": {"vi": "sụp đổ / xé lòng / nhói lòng", "category": "Douyin/Bilibili", "source": "Auto-Sync"},
    "绝绝子": {"vi": "đỉnh kout / hết nước chấm / mười điểm", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "硬核": {"vi": "xịn xò / bá đạo / khét lẹt", "category": "Bilibili", "source": "Auto-Sync"},
    "显眼包": {"vi": "thánh gây chú ý / chúa tấu hề", "category": "Douyin", "source": "Auto-Sync"},
    "YYDS": {"vi": "mãi đỉnh / Vạn Tuế Đấu Thần", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "芭比Q了": {"vi": "toang rồi / xong đời / BBQ rồi", "category": "Douyin", "source": "Auto-Sync"},
    "泰裤辣": {"vi": "quá ngầu / bá cháy / quá khét", "category": "Douyin", "source": "Auto-Sync"},
    "栓Q": {"vi": "cảm ơn nhiều (mỉa mai) / Thank You", "category": "Douyin", "source": "Auto-Sync"},
    "特种兵式": {"vi": "kiểu lính đặc nhiệm / thần tốc", "category": "Douyin", "source": "Auto-Sync"},
    "沉浸式": {"vi": "kiểu đắm chìm / trải nghiệm thực tế", "category": "Douyin/Bilibili", "source": "Auto-Sync"},
    "搭子": {"vi": "cạ cứng / bạn đồng hành / cạ ăn cạ chơi", "category": "Xiaohongshu", "source": "Auto-Sync"},
    "情绪价值": {"vi": "giá trị cảm xúc / năng lượng tích cực", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "特种兵旅游": {"vi": "du lịch thần tốc / đi như chạy giặc", "category": "Douyin", "source": "Auto-Sync"},
    "尊嘟假嘟": {"vi": "thật á / đùa à / thật không đó", "category": "Douyin", "source": "Auto-Sync"},
    "主打一个": {"vi": "chủ yếu là / chuẩn bài / nhấn mạnh", "category": "Bilibili", "source": "Auto-Sync"},
    "纯爱战士": {"vi": "chiến thần thuần ái / yêu chân thành", "category": "Bilibili", "source": "Auto-Sync"},
    "牛马": {"vi": "trâu ngựa / kiếp làm thuê / kiếp cày cuốc", "category": "Douyin", "source": "Auto-Sync"},
    "恋爱脑": {"vi": "não yêu đương / lụy tình / mù quáng", "category": "Xiaohongshu", "source": "Auto-Sync"},
    "打工人": {"vi": "dân làm thuê / kiếp cày tiền", "category": "Douyin/Bilibili", "source": "Auto-Sync"},
    "社死": {"vi": "xấu hổ muốn độn thổ / chết xã hội", "category": "Douyin", "source": "Auto-Sync"},
    "社恐": {"vi": "sợ giao tiếp / hướng nội sợ đám đông", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "社牛": {"vi": "thánh giao tiếp / thần thái tự tin", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "干饭人": {"vi": "thánh ăn / thần đồ ăn / nhiệt huyết ăn uống", "category": "Douyin", "source": "Auto-Sync"},
    "内卷": {"vi": "cuốn nội bộ / áp lực cạnh tranh khốc liệt", "category": "Bilibili", "source": "Auto-Sync"},
    "躺平": {"vi": "nằm phẳng / mặc kệ đời / buông xuôi", "category": "Bilibili/Douyin", "source": "Auto-Sync"},
    "润": {"vi": "chuồn / trốn / vọt lẹ", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "吃瓜": {"vi": "hóng hớt / hóng drama / ăn dưa", "category": "Douyin/Bilibili", "source": "Auto-Sync"},
    "嗑CP": {"vi": "chèo thuyền CP / đẩy thuyền cặp đôi", "category": "Bilibili", "source": "Auto-Sync"},
    "凡尔赛": {"vi": "khoe khoang tinh tế / giả vờ khiêm tốn", "category": "Douyin", "source": "Auto-Sync"},
    "夺笋": {"vi": "độc mồm độc miệng / mất nết / đào măng", "category": "Douyin", "source": "Auto-Sync"},
    "伤害性不高": {"vi": "sát thương không cao nhưng nhục nhã", "category": "Douyin", "source": "Auto-Sync"},
    "侮辱性极强": {"vi": "tính xúc phạm cực kỳ cao", "category": "Douyin", "source": "Auto-Sync"},
    "xswl": {"vi": "cười chết mất / cười sặc sụa", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "zqsg": {"vi": "chân tình thực cảm / lòng thành", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "srds": {"vi": "tuy nhiên / mặc dù vậy", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "nsdd": {"vi": "bạn nói đúng rồi đấy / chuẩn luôn", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "u1s1": {"vi": "có một nói một / nói thật lòng", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "pyq": {"vi": "vòng bạn bè / bảng tin Moments", "category": "WeChat", "source": "Auto-Sync"},
    "plmm": {"vi": "em gái xinh đẹp / mỹ nữ", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "xgg": {"vi": "anh trai đẹp trai / soái ca", "category": "Trend Giới Trẻ", "source": "Auto-Sync"},
    "抖人": {"vi": "dân quạt Douyin / cư dân mạng Douyin", "category": "Douyin", "source": "Auto-Sync"},
    "小红书": {"vi": "Tiểu Hồng Thư / RED book", "category": "Xiaohongshu", "source": "Auto-Sync"},
    "种草": {"vi": "gieo mầm mê mẩn / muốn mua ngay", "category": "Xiaohongshu", "source": "Auto-Sync"},
    "拔草": {"vi": "nhổ cỏ / hết mê / hủy mua", "category": "Xiaohongshu", "source": "Auto-Sync"},
    "踩雷": {"vi": "dẫm phải mìn / mua trúng đồ tệ", "category": "Xiaohongshu", "source": "Auto-Sync"},
    "避雷": {"vi": "né mìn / khuyên nên tránh xa", "category": "Xiaohongshu", "source": "Auto-Sync"},
    "硬控": {"vi": "khống chế cứng / xem mê không dứt ra được", "category": "Douyin", "source": "Auto-Sync"},
    "卡拉米": {"vi": "kẻ vô danh / tép riêu", "category": "Douyin", "source": "Auto-Sync"},
    "小老弟": {"vi": "em trai nhỏ / chú em", "category": "Douyin", "source": "Auto-Sync"},
    "老铁": {"vi": "anh em chí thiết / cạ cứng", "category": "Douyin", "source": "Auto-Sync"}
}

def sync_online_trending_words(custom_url=None):
    """Đồng bộ danh sách từ lóng / hot words mới nhất từ các kho công khai trực tuyến."""
    manager = TrendingSlangManager()

    sources = [
        custom_url,
        "https://cdn.jsdelivr.net/gh/chatgpt-prompts/chinese-slang@main/trending_dict.json"
    ]

    online_slangs = dict(EXPANDED_TRENDING_SLANGS)

    fetched = False
    for url in sources:
        if not url:
            continue
        try:
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            online_slangs[k] = v
                        elif isinstance(v, str):
                            online_slangs[k] = {"vi": v, "category": "Douyin/Bilibili", "source": "Auto-Sync"}
                    fetched = True
                    break
        except Exception:
            pass

    added_count = manager.merge_dict(online_slangs, default_source="Auto-Sync", overwrite_user_custom=False)
    manager.load_dict()
    
    return {
        "success": True,
        "added_count": added_count,
        "total_count": len(manager.slang_dict),
        "fetched_online": fetched
    }
