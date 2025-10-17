# 웹 API 활용
# 그래프를 pdf파일로 저장

import requests
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

apikey = "b1439a93d08b8cc8581dc60f3875e18d"

cities = ["Seoul", "New York", "Tokyo", "Sydney"]

api = "https://api.openweathermap.org/\
data/2.5/weather?q={city}&APPID={key}"

k2c = lambda k: k - 273.15

temp_min_list = []
temp_max_list = []

for i in range(len(cities)):
    url = api.format(city=cities[i], key=apikey)
    r = requests.get(url)
    # print(r.text)
    data = json.loads(r.text)
    print(data)
    temp_min_list.append(k2c(data["main"]["temp_min"]))
    temp_max_list.append(k2c(data["main"]["temp_max"]))

# print("+ 도시 =", data["name"])
# print("| 날씨 =", data["weather"][0]["description"])
# print("| 최저 기온 =", k2c(data["main"]["temp_min"]))
# print("| 최고 기온 =", k2c(data["main"]["temp_max"]))
# print("| 습도 =", data["main"]["humidity"])
# print("| 기압 =", data["main"]["pressure"])
# print("| 풍향 =", data["wind"]["deg"])
# print("| 풍속 =", data["wind"]["speed"])
# print("")

plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams['axes.unicode_minus'] =False

plt.scatter(cities, temp_min_list, color='blue',label='최저 기온', s=100, marker='o')
plt.title('도시별 최저 기온')
plt.xlabel('도시')
plt.ylabel('기온 (°C)')
plt.ylim(-15, 30)
plt.xlim(-0.5, len(cities) - 0.5)
plt.grid(True)
plt.legend()
plt.tight_layout()

with PdfPages(r'.\python\ch23\task\example.pdf') as pdf:
    pdf.savefig()
    plt.show()