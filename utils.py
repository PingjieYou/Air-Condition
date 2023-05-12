import csv
import sys
import torch
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# pd.set_option('display.max_rows', None) # 展示所有行
# pd.set_option('display.max_columns', None) # 展示所有列


def url2soup(url, headers=None):
    '''通过BeautifulSoup获取url的html'''
    url = url.replace('\r', '').replace('\n', '')  # 消除特殊字符对爬虫的影响
    response = requests.get(url, headers=headers)  # GET
    response.encoding = 'gb2312'  # 页面编码
    text = response.text  # 获取html的文本信息
    soup = BeautifulSoup(text, 'html.parser')  # 使用soup解析html

    return soup


def df2csv(df, csv_path):
    '''保存csv文件'''
    df.to_csv(csv_path)


def csv2df(csv_path):
    '''读取csv'''
    df = pd.read_csv(csv_path, header=None, index_col=0)
    return df


def txt2np(aqi_and_lish_txt):
    '''存url的txt文件转为np数组'''
    aqi_txt, lish_txt = aqi_and_lish_txt

    aqi_list = np.loadtxt(aqi_txt, dtype=str)
    lish_list = np.loadtxt(lish_txt, dtype=str)

    return aqi_list, lish_list


def append2csv(csv_path, data):
    '''向csv中追加数据'''
    f = open(csv_path, "a", encoding="utf-8", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(data)
    f.close()


def createMysqlTable(db):
    '''创建mysql表'''
    cursor = db.cursor()  # 创建游标
    cursor.execute(
        'create table weather(times varchar(100),city varchar(100),date varchar(100),date_ varchar(100),quality varchar(100),aqi varchar(100),aqi_rank varchar(100),pm25 varchar(100),pm10 varchar(100),so2 varchar(100),no2 varchar(100),co varchar(100),o3 varchar(100),situation varchar(100),wind varchar(100), tem varchar(100))'
    )


def insert2mysql(db, data):
    '''将data数据插入进数据库'''
    sql = 'insert into weather (times,city,date,date_,quality,aqi,aqi_rank,pm25,pm10,so2,no2,co,o3,situation,wind,tem) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
    cursor = db.cursor()  # 创建游标
    cursor.execute(sql, tuple(data))


def parse_url(aqi_url, lish_url, save_path, db=None, url_header="http://www.tianqihoubao.com"):
    '''解析AQI和历史天气url'''
    data = []
    skip = 1e7  # 跳过skip以下的
    end = 1e7  # 结束位置

    for index, (aqi, lish) in enumerate(zip(aqi_url, lish_url)):
        # 跳转到skip位置
        if index < skip and skip < len(aqi_url):
            continue
        if index == end:
            break
        # 补全url
        aqi = url_header + aqi
        lish = url_header + lish
        soup_aqi = url2soup(aqi)
        soup_lish = url2soup(lish)
        # 定位细节标签
        table = soup_aqi.find_all('table')
        name = soup_aqi.find_all('div', {'class', 'wdetail'})[0].find_all('h1')
        name = name[0].text.strip()
        name_start = name.find('月')
        name_end = name.find('空气')
        name = name[name_start + 1:name_end]
        table_lish = soup_lish.find_all('table')
        print("Times:", str(index), "，正在解析的城市名称:", name)
        print('解析AQI的url', aqi)
        print('解析历史天气的url', lish, '\n')

        try:
            # 解析AQI行数据
            for i in range(len(table[0].find_all('tr'))):
                td_all = table[0].find_all('tr')[i].find_all('td')  # 解析每行的数据
                # city = month_item[0][8:10]  # 城市名字
                city = name  # 城市名字
                date = td_all[0].text.strip()  # 日期
                quality = td_all[1].text.strip()  # 空气质量
                aqi = td_all[2].text.strip()  # aqi指标
                aqi_rank = td_all[3].text.strip()  # aqi排名
                pm25 = td_all[4].text.strip()  # pm2.5
                pm10 = td_all[5].text.strip()  # pm1.0
                so2 = td_all[6].text.strip()  # So2
                no2 = td_all[7].text.strip()  # No2
                co = td_all[8].text.strip()  # CO
                o3 = td_all[9].text.strip()  # O3
                td_all = table_lish[0].find_all(
                    'tr')[i].find_all('td')  # 解析每行的数据
                date_ = td_all[0].text.strip().replace(' ', '')
                suituaion = td_all[1].text.strip().replace(" ", "")
                tem = td_all[2].text.strip()
                wind = td_all[3].text.strip()
                append2csv('csv/chongqing.csv', [city, date, date_, quality, aqi, aqi_rank,
                                                 pm25, pm10, so2, no2, co, o3, suituaion, wind, tem, index])
                if db != None:
                    insert2mysql(db=db, data=[(city), (date), (date_), (quality), (aqi), (aqi_rank),
                                              (pm25), (pm10), (so2), (no2), (co), (o3), (suituaion), (wind), (tem), (index)])
        except:
            pass


