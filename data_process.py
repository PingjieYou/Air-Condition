from sklearn.preprocessing import StandardScaler

index2name = {0: 'times', 1: 'city', 2: 'date', 3: 'date_', 4: 'aqi', 5: 'aqi_rank', 6: 'pm25', 7: 'pm10',
              8: 'so2', 9: 'no2', 10: 'co', 11: 'o3', 12: 'situation', 13: 'wind', 14: 'temp'}
name2index = {'times': 0, 'city': 1, 'date': 2, 'date_': 3, 'quality': 4, 'aqi': 5, 'aqi_rank': 6, 'pm25': 7, 'pm10': 8,
              'so2': 9, 'no2': 10, 'co': 11, 'o3': 12, 'situation': 13, 'wind': 14, 'temp': 15, 'min_temp': 16,
              'max_temp': 17}
name2index_chongqing = {'city': 0, 'date': 1, 'date_': 2, 'quality': 3, 'aqi': 4, 'aqi_rank': 5, 'pm25': 6, 'pm10': 7,
                        'so2': 8, 'no2': 9, 'co': 10, 'o3': 11, 'situation': 12, 'wind': 13, 'temp': 14, 'times': 15,
                        'min_temp': 16, 'max_temp': 17}


def data_cleaning(df, name2index=name2index):
    '''数据清洗'''
    # 删除无效数据
    row_indexs = df[df[name2index['date']] == '日期'].index
    df.drop(row_indexs, inplace=True)
    # 删除天气情况的无效字符
    df[name2index['situation']] = df[name2index['situation']
                                     ].str.replace('\r', '')
    df[name2index['situation']] = df[name2index['situation']
                                     ].str.replace('\n', '')
    # 删除风力的无效字符
    df[name2index['wind']] = df[name2index['wind']].str.replace('\r', '')
    df[name2index['wind']] = df[name2index['wind']].str.replace('\n', '')
    df[name2index['wind']] = df[name2index['wind']].str.replace(' ', '')
    # 删除温度的无效数据
    df[name2index['temp']] = df[name2index['temp']].str.replace('\r', '')
    df[name2index['temp']] = df[name2index['temp']].str.replace('\n', '')
    df[name2index['temp']] = df[name2index['temp']].str.replace(' ', '')


def data_sort(df, name2index=name2index):
    '''数据按月份排序'''
    df = df.sort_values(by=name2index['city'], ascending=False)
    df = df.sort_values(by=name2index['date'], ascending=False)
    return df


def quantization(df, name2index=name2index):
    '''将属性值量化'''
    # 量化aqi
    aqi_values_list = sorted(list(set(df[name2index['quality']].values)))
    aqi_values_dict = {k: v for k, v in zip(
        aqi_values_list, range(len(aqi_values_list)))}
    for aqi_value in aqi_values_list:
        df.loc[df[name2index['quality']] == aqi_value,
               name2index['quality']] = aqi_values_dict[aqi_value]
    # 添加最低和最高气温
    df_temp_index = df[name2index['temp']].str.find('/')
    df_min_temp = []
    df_max_temp = []
    for i in range(len(df_temp_index)):
        df_min_temp.append(df.iloc[i][name2index['temp']]
                           [:df_temp_index.values[i] - 1])
        df_max_temp.append(df.iloc[i][name2index['temp']]
                           [df_temp_index.values[i] + 1:-1])
    df[name2index['min_temp']] = df_min_temp
    df[name2index['max_temp']] = df_max_temp


def get_data_by_city(df, city_name, name2index=name2index):
    '''按城市名获取所有数据'''
    df_city = df.loc[df[name2index['city']] == city_name]
    return df_city


def div_data_by_month_and_year(df, month_start=1, month_end=12, year_start=13, year_end=22, name2index=name2index):
    '''按年月划分数据存入到list中'''
    dic = {}
    monthes = ['01', '02', '03', '04', '05',
               '06', '07', '08', '09', '10', '11', '12']
    years = [num for num in range(year_start, year_end + 1)]
    for year in years:
        for month in monthes:
            df_year = df.loc[df[name2index['date']].str[2:4] == str(year)]
            df_month = df_year.loc[df[name2index['date']].str[5:7] == month]
            if len(df_month) != 0:
                dic[str(year) + "-" + month] = df_month
    return dic


def get_temp_data(df, name2index=name2index):
    '''获取numpy格式的温度数据'''
    temp_data = df.values[1:]
    return temp_data[:, -2:]


def get_aqi_data(df, name2index=name2index):
    '''获取numpy格式的aqi数据'''
    aqi_data = df.values[1:]
    return aqi_data[:, 3:-5]


def get_city_names(df, name2index=name2index):
    '''获取所有城市的名称'''
    city_names = list(set(df[name2index['city']].values[1:]))
    return city_names

def get_cls_data(df,name2index=name2index):
    '''获取分类数据'''
    attr = ['quality','pm25','pm10','so2','no2','co','o3']
    df_attr_values = df[[name2index[name] for name in attr]].values
    x = df_attr_values[:,1:]
    y = df_attr_values[:,0]
    return x,y

def get_rgs_data(df,name2index=name2index):
    '''获取回归数据'''
    df_values = df[[name2index['min_temp'],name2index['max_temp']]].values
    
    return df_values

def pre_processing(data):
    """
    预处理

    :param data: 空气质量数据
    :return:
    """
    standard_scaler = StandardScaler()
    standard_scaler.fit(data)
    data = standard_scaler.transform(data)

    return data