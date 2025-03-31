import requests
import pandas as pd

API_KEY = "0a9c1efb66e7fc8074116128f8de0c5d"  # 请替换为您的高德 API Key
city = "香港"
keywords = ["养老院", "护老院", "疗养院", "老人院"]
all_places = []

for keyword in keywords:
    page = 1
    while True:
        url = f"https://restapi.amap.com/v3/place/text?keywords={keyword}&city={city}&offset=50&page={page}&key={API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "pois" not in data or not data["pois"]:
            break

        for poi in data["pois"]:
            name = poi["name"]
            location = poi["location"].split(",")
            lon, lat = location[0], location[1]
            address = poi.get("address", "未知")
            all_places.append([keyword, name, lat, lon, address])

        print(f"已获取关键词 '{keyword}' 的第 {page} 页数据，共 {len(data['pois'])} 条")
        page += 1

# 存入 Excel
df = pd.DataFrame(all_places, columns=["类别", "名称", "经度", "纬度", "地址"])
df.to_excel("香港养老相关机构.xlsx", index=False)

print("done")
