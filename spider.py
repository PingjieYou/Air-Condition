from audioop import reverse
from email import header
import os
import numpy as np
import pandas as pd
from utils import url2soup

# 避开反扒机制，人工爬取
# 各个url
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
}

url_header = "http://www.tianqihoubao.com"
url_aqi = "http://www.tianqihoubao.com/aqi/"
url_lish = "http://www.tianqihoubao.com/lishi/"
data = []
cities = []
cities_aqi = []
cities_lish = []
aqi_list = []
lish_list = []
url_city = []
url_month = []
url_lish_month = []
url_lish_city = []

# 获取各个城市的url和天气url
city_soup = url2soup(url_aqi, headers)
city_soup_dl = city_soup.find_all("dl")
city_lishi_soup = url2soup(url_lish, headers)
city_lishi_soup = city_lishi_soup.find_all("dl")

# 获取各个城市aqi的url
for dl in city_soup_dl:
    item = dl.find_all("a")
    for i in range(len(item)):
        # 去除表中重复的值
        if not (dl == city_soup_dl[0] and item[i].text.strip() in ['广州', '杭州', '成都', '深圳', '全国空气质量排名']):
            cities_aqi.append(item[i].text.strip())
            # cities_aqi.append(item[i].text.strip())

# 获取各个城市历史天气的url
for dl in city_lishi_soup:
    item = dl.find_all("a")
    for i in range(len(item)):
        if not (dl == city_soup_dl[0]) and item[i]['href'][-3:] != 'htm':
            cities_lish.append(item[i].text.strip())
            # url_lish_city.append([item[i].text.strip(), item[i]["href"]])

cities = list(set(cities_aqi).intersection(set(cities_lish)))  # 城市取并集

# 获取各个城市aqi的url
for dl in city_soup_dl:
    index = 0
    item = dl.find_all("a")
    for i in range(len(item)):
        # 去除表中重复的值
        if not (dl == city_soup_dl[0] and item[i].text.strip() in ['广州', '杭州', '成都', '深圳', '全国空气质量排名']):
            if item[i].text.strip() in cities and item[i]['href'][-14:] != 'hljgannan.html':
                url_city.append([item[i].text.strip(), item[i]['href']])
url_city = sorted(url_city, key=lambda x: x[:][0])

# 获取各个城市历史天气的url
for dl in city_lishi_soup:
    item = dl.find_all("a")
    for i in range(len(item)):
        if not (dl == city_soup_dl[0]) and item[i]['href'][-3:] != 'htm' and item[i].text.strip() in cities:
            url_lish_city.append([item[i].text.strip(), item[i]["href"]])
url_lish_city = sorted(url_lish_city, key=lambda x: x[:][0])

# -----------------------
#     爬取重庆的数据
# -----------------------
url_aqi_chongqing = [['重庆', '/aqi/chongqing.html']]
url_lish_chongqing = [['重庆', '/lishi/chongqing.html']]

start_index = 0
end_index = 100000
# 获取城市每个月份的aqi的url
# for i, (city_item, city_lish_item) in enumerate(zip(url_city, url_lish_city)):
for i, (city_item, city_lish_item) in enumerate(zip(url_aqi_chongqing, url_lish_chongqing)):
    if start_index <= i < end_index:
        city_url = url_header + city_item[1]  # 完整的城市空气质量url
        city_lish_url = url_header + city_lish_item[1]  # 完整城市历史天气url
        city_soup = url2soup(city_url)
        city_lish_soup = url2soup(city_lish_url)
        city_soup_div = city_soup.find_all('div', {'class', 'box p'})
        city_lish_soup_div = city_lish_soup.find_all(
            'div', {'class', 'box pcity'})[:-2]
        city_soup_div = list(reversed(city_soup_div))
        city_lish_soup_div = list(reversed(city_lish_soup_div))
        print("正在爬取第", str(i + 1), "个数据，AQIurl, city:",
              city_item[0], ', AQI url:', city_url, ', Weather url:', city_lish_url)

        # 获取每个月的标签
        for month in city_soup_div[0].find_all('a'):
            # 只取2016~2022年的数据
            if month['title'][:4] == '2012':
                break
            url_month.append([month["title"], month['href']])
            aqi_list.append(month['href'])

        # 获取每个月的标签
        for k, month_index in enumerate(range(len(city_lish_soup_div))):
            month_temp = []
            # for ele in city_soup_div[0].find_all('a'):
            #     print(ele.text.strip())
            for month_ in city_lish_soup_div[k].find_all('a'):
                month_temp.append(
                    [month_.text.strip()[:], month_['href']])  # 添加临时url
            # 反转后添加到url_lish
            for ele in reversed(month_temp):
                url_lish_month.append(ele)
                lish_list.append(ele[1])
            # 只遍历k个年份
            if k == 100:
                break
    # 只遍历三个城市
    # if i == 1:
    #     break
np.savetxt("./csv/aqi_url" + str(start_index) +
           ".txt", np.array(aqi_list), fmt="%s")
np.savetxt("./csv/weather_url" + str(start_index) + ".txt",
           np.array(lish_list), fmt="%s")
