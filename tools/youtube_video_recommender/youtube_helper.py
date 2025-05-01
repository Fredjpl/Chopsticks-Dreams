import os
from googleapiclient.discovery import build
import pandas as pd
import json

def search_youtube_recipes(dish_name: str, max_results: int = 5):
    api_key = os.environ.get("GOOGLE_API")
    if not api_key:
        raise EnvironmentError("Environment variable 'GOOGLE_API' not set.")

    youtube = build('youtube', 'v3', developerKey=api_key)
    query = f"{dish_name} cooking tutorial"

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
