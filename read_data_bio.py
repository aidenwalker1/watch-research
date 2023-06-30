import math
import json
import interactions as inter
from datetime import datetime
import stat_features
import numpy as np
import csv

def read_data(path, responses, interactions, read_range, home_coords) :
    file = open(path, 'r')

    file.readline()
    st = None

    read_index = 0

    data = []
    min_data = []
    all_data = []

    heartrate = 70

    rotation = 0
    acceleration = 0

    lat = home_coords[0]
    long = home_coords[1]

    (xr, yr, zr) = (0,0,0)
    (xa, ya, za) = (0,0,0)
    (yaw, pitch, roll) = (0,0,0)

    last_response = 0
    last_time = None
    response_index = 0
    # get 30 minutes of data
    # get averages from this period
    # 

    while st != '' and read_index < read_range:
        st = file.readline()
        try :
            fields = json.loads(st[:st.find(',\n')])
        except :
            break

        if fields['message_type'] == None :
            if 'heart' in st :
                heartrate = fields['sensors']['heart_rate']
                continue
        
        if fields['message_type'] == 'location' :
            sensor = fields['sensors']
            lat = float(sensor['latitude'])
            long = float(sensor['longitude'])
            lat = round(lat, 2)
            long = round(long, 2)
            continue     

        stamp = str(fields["stamp"])  
        stamp = stamp[:stamp.index("+")]

        time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
        if last_time == None :
            last_time = time

        response_index, time_left = inter.in_interval(time, responses, last_response)
        sensor = fields["sensors"]

        if response_index != last_response :
            if len(min_data) < 26 :
                print('bad')
            else :
                all_data.append((min_data[0:26], last_response))
            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)
            last_time = time
            data = []
            min_data = []
            last_response = response_index

        #gets current rotation
        xr += sensor['rotation_rate_x']
        yr += sensor['rotation_rate_y']
        zr += sensor['rotation_rate_z']

        #gets current rotation
        xa += sensor['user_acceleration_x']
        ya += sensor['user_acceleration_y']
        za += sensor['user_acceleration_z']

        if len(data) == 60 :
            min_data.append(analyze_data(data,interactions[response_index], response_index))
            data = []

        if (time-last_time).total_seconds() >= 1 :
            rotation = math.pow(xr**2 + yr**2 + zr**2, 1/2)
            acceleration = math.pow(xa**2 + ya**2 + za**2, 1/2)
            last_time = time
            yaw = sensor['yaw']
            pitch = sensor['pitch']
            roll = sensor['roll']
            #data.append([xr, yr, zr, xa, ya, za, heartrate, close_loc(lat, long, home_coords[0], home_coords[1])])

            data.append([rotation, acceleration, heartrate, yaw, pitch, roll, close_loc(lat, long, home_coords[0], home_coords[1])])

            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)

            read_index += 1
            print(read_index)
        
    if response_index == last_response :
        if len(min_data) < 26 :
                print('bad')
        else :
            all_data.append((min_data[0:26], last_response))
    
    file.close()
    return all_data

def analyze_data(data, interaction, index) :
    new_data = []
    data = np.array(data)
    r = data[:,0]
    a = data[:,1]
    h = data[:,2]
    y = data[:,3]
    p = data[:,4]
    ro = data[:,5]
    lat = data[:,6]
    long = data[:,7]
    interaction = np.array(interaction)
    loc = data[:,8]
    s1 = stat_features.generate_statistical_features(r)
    s2 = stat_features.generate_statistical_features(a)
    s3 = stat_features.generate_statistical_features(h)
    s4 = stat_features.generate_statistical_features(interaction)
    s5 = stat_features.generate_statistical_features(y)
    s6 = stat_features.generate_statistical_features(p)
    s7 = stat_features.generate_statistical_features(ro)
    s8 = stat_features.generate_statistical_features(lat)
    s9 = stat_features.generate_statistical_features(long)
    #s5 = 
    #new_data += ([s1] + [s2] + [s3])
    new_data += (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + [np.average(loc)])

    return new_data

def find_home(path, read_range) :
    file = open(path, 'r')

    st = None

    freq = dict()
    file.readline()
    index = 0
    while st != '' and index < read_range * 5 :
        st = file.readline()

        fields = json.loads(st[:st.find(',\n')])

        if fields['message_type'] == 'location' :
            sensor = fields['sensors']
            lat = float(sensor['latitude'])
            long = float(sensor['longitude'])
            lat = round(lat, 2)
            long = round(long, 2)

            if freq.get((lat,long)) == None :
                freq[(lat,long)] = 1
            else :
                freq[(lat,long)] += 1
        index += 1
    file.close()

    freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)
    if len(freq) == 0 :
        return 0,(0,0)
    return freq, freq[0][0]


def calculate_bearing(lat1, long1, lat2, long2):
    d1 = long2 - long1
    y = math.sin(d1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(d1)
    bearing = math.atan2(y, x)
    bearing = np.rad2deg(bearing)
    return bearing

def calculate_distance(lat1, long1, lat2, long2):
    d1 = (lat2 - lat1) * (lat2 - lat1)
    d2 = (long2 - long1) * (long2 - long1)
    return np.sqrt(d1 + d2)

def close_loc(lat1, long1, lat2, long2):
    """ Determine if one location is within threshold distance to another.
    """
    distance = calculate_distance(lat1, long1, lat2, long2)
    if distance < 0.015:
        return 1
    else:
        return 0
    

def get_y(responses, data, y_index) :
    return [int(responses[data[i][-1]][y_index]) for i in range(0,len(data))]

def get_response_value(responses, response_indices, wanted_label) :
    return [int(responses[r][wanted_label]) for r in response_indices]

def read_computed(path) :
    file = open(path, 'r')
    
    csv_reader = csv.reader(file)

    data = []
    hourly = []
    all_hour = []
    daily = []
    all_daily = []
    labels = []
    x = []
    y = 0
    for row in csv_reader :
        r = [float(s) for s in row]
        if float(row[-2]) > 1800 :
            if len(x) > 26 :
                data.append(x[0:26])
                labels.append(y)
                x = []
        else :
            x.append(r[:-2])
            y = int(r[-1])
        hourly.append(r[:-2])
        daily.append(r[:-2])

        if len(hourly) == 60 :
            cur = []
            hourly = np.array(hourly)
            for i in range(0,hourly.shape[1] - 30,30) :
                cur += stat_features.generate_statistical_features(hourly[:,i])
            cur += [r[-1]]
            all_hour.append(cur)
            hourly = []
        if len(daily) == 1440 :
            cur = []
            daily = np.array(daily)
            for i in range(0,daily.shape[1] - 30,30) :
                cur += stat_features.generate_statistical_features(daily[:,i])
            cur += [r[-1]]
            all_daily.append(cur)
            daily = []

    return data, labels, all_daily, all_hour
    #return np.array(data)

def build_data(computed, labels, hourly, daily, responses) :
    alls = []
    cur_index = 0
    cur_day = 0
    for i in range(len(computed)) :
        c = computed[i]
        cur = []
        
        while cur_index < len(hourly) and hourly[cur_index][-1] <= labels[i] :
            cur_index += 1

        while cur_day < len(daily) and daily[cur_day][-1] <= labels[i] :
            cur_day += 1
            
        if cur_index < 4 or cur_day < 1 :
            continue

        hours = hourly[cur_index-5:cur_index-1]
        for i in range(len(hours)) :
            hours[i] = hours[i][:-1]
        day = daily[cur_day-1]
        day = day[:-1]
        cur.append(c)
        cur.append(hours)
        cur.append(day)
        cur.append(responses[i][1:])

        alls.append(cur)
    
    return alls

def compute_hourly(data) :
    data = np.array(data)
    hourly = []
    for i in range(0, data.shape[0]) :
        cur = data[i]
        l = []
        for j in range(0, data.shape[1]) :
            d = cur[j]
            l.append(stat_features.generate_statistical_features(d))
            

    pass