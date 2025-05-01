from googleapiclient.discovery import build

def search_youtube_recipes(dish_name: str, api_key: str, max_results: int = 5):
    """
    用于搜索与指定菜名相关的做菜教学视频。
    
    参数：
    - dish_name (str): 菜名，例如 "泰国绿咖喱"
    - api_key (str): 你的 YouTube Data API 密钥
    - max_results (int): 返回的最大视频数量，默认5个

    返回：
    - List[Dict]: 包含视频标题和链接的列表
    """
    youtube = build('youtube', 'v3', developerKey=api_key)

    # 构造搜索关键词
    query = f"{dish_name} 做饭 教学"

    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results,
        order="relevance"
    )
    response = request.execute()

    results = []
    for item in response['items']:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        results.append({'title': title, 'url': video_url})

    return results
