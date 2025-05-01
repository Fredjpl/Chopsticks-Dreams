import requests
import json
import os

api_key = os.environ.get("GOOGLEMAP_API")
if not api_key:
    raise EnvironmentError("Environment variable 'GOOGLEMAP_API' not set.")

def get_lat_lng_from_zip(zipcode):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zipcode}&key={api_key}"
    response = requests.get(url)
    res_json = response.json()
    if res_json['status'] == 'OK':
        location = res_json['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        raise ValueError(f"Can't locate zipcode {zipcode}. Error：{res_json['status']}")

def search_grocery_store_nearby(zipcode, item_list, radius=3000):
    lat, lng = get_lat_lng_from_zip(zipcode)
    results_by_item = {}

    for item in item_list:
        url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"location={lat},{lng}&radius={radius}&keyword={item}+grocery&key={api_key}"
        )
        response = requests.get(url)
        res_json = response.json()
        results = []
        for place in res_json.get('results', []):
            results.append({
                'name': place['name'],
                'address': place.get('vicinity'),
                "lat"      : place["geometry"]["location"]["lat"],
                "lng"      : place["geometry"]["location"]["lng"],
                'open_now': place.get('opening_hours', {}).get('open_now', 'Unknown'),
            })
        results_by_item[item] = results[:5]

    return results_by_item