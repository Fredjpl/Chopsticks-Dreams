import requests

def get_user_location():
    """
    基于公网 IP 获取用户的大致位置（城市 + 经纬度）

    返回:
    - loc: str 类型经纬度 "latitude,longitude"
    - city: 城市名
    """
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()
        loc = data.get("loc")  # e.g., "37.7749,-122.4194"
        city = data.get("city", "")
        return loc, city
    except Exception as e:
        print("无法获取用户位置：", e)
        return None, None
    
def search_nearby_stores(api_key: str, location: str, keyword: str, radius: int = 1500):
    """
    使用 Google Places API 查找附近出售指定食材的商店

    参数:
    - api_key: Google Maps API Key
    - location: 用户位置（纬度,经度）例如 "37.7749,-122.4194"
    - keyword: 搜索关键词（通常是缺失食材）
    - radius: 搜索半径（单位：米）

    返回:
    - 商店信息列表
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": location,
        "radius": radius,
        "keyword": keyword + " grocery",
        "type": "grocery_or_supermarket",
        "key": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    results = []
    for place in data.get("results", []):
        store = {
            "name": place.get("name"),
            "address": place.get("vicinity"),
            "rating": place.get("rating"),
            "open_now": place.get("opening_hours", {}).get("open_now", "未知")
        }
        results.append(store)

    return results
