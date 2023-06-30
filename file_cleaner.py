import json
from datetime import datetime
import interactions as inter
import responses as resp
import math
import read_data_bio

def clean_bt(path, responses, dyad) :
    file = open(path, 'r')
    cleaned_data = open(dyad + '_clean_bt.json', 'w')
    st = file.readline()
    last_index = 0

    

    while st != '':
        st = file.readline()
        try:
            fields = json.loads(st[:st.find(',\n')])
        except:
            break
        
        if fields['sensors'].get('peer_bt_rssi') == None :
            continue

        stamp = str(fields["stamp"])
        stamp = stamp[:stamp.index("+")]
        time = None
        if '.' in stamp :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
        else :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
            index = st.index("+")
            st = st[:index] + ".0" + st[index:]

        response_index = inter.in_interval(time, responses, last_index)

        if response_index == -1:
            continue
        if response_index != last_index :
            last_index = response_index
        cleaned_data.write(st)
    file.close()
    cleaned_data.close()

def clean_data(path, responses, dyad) :
    file = open(path, 'r')
    cleaned_data = open(dyad + '_clean_sensor_data.json', 'w')
    st = file.readline()
    last_index = 0

    while st != '':
        st = file.readline()

        fields = json.loads(st[:st.find(',\n')])

        if fields['message_type'] == None :
            if 'heart' in st :
                cleaned_data.write(st)
                continue

        if fields['message_type'] == 'location' :
            cleaned_data.write(st)
            continue

        if fields['message_type'] != 'device_motion' :
            continue

        stamp = str(fields["stamp"])
        stamp = stamp[:stamp.index("+")]
        time = None
        if '.' in stamp :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
        else :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
            index = st.index("+")
            st = st[:index] + ".0" + st[index:]

        response_index = inter.in_interval(time, responses, last_index)

        if response_index == -1:
            continue
        if response_index != last_index :
            last_index = response_index
        cleaned_data.write(st)
    file.close()
    cleaned_data.close()

def compute_data(path, responses, interactions, home_coords, dyad) :
    file = open(path, 'r')
    cleaned_data = open(dyad + '_computed.csv', 'w')
    st = file.readline()

    lat = home_coords[0]
    long = home_coords[1]

    (xr, yr, zr) = (0,0,0)
    (xa, ya, za) = (0,0,0)
    (yaw, pitch, roll) = (0,0,0)

    last_index = 0
    last_time = None
    response_index = 0
    read_index = 0
    data = []
    time = None

    rotation = 0
    acceleration = 0
    heartrate = 80

    while st != '':
        st = file.readline()

        try :
            fields = json.loads(st[:st.find(',\n')])
        except:
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

        if fields['message_type'] != 'device_motion' :
            continue

        stamp = str(fields["stamp"])
        stamp = stamp[:stamp.index("+")]

        if '.' in stamp :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
        else :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
            index = st.index("+")
            st = st[:index] + ".0" + st[index:]

        if last_time == None :
            last_time = time

        response_index, time_left = inter.in_interval(time, responses, last_index)

        sensor = fields["sensors"]

        if response_index !=-1 and response_index != last_index :
            last_index = response_index

        #gets current rotation
        xr += sensor['rotation_rate_x']
        yr += sensor['rotation_rate_y']
        zr += sensor['rotation_rate_z']

        #gets current rotation
        xa += sensor['user_acceleration_x']
        ya += sensor['user_acceleration_y']
        za += sensor['user_acceleration_z']

        if len(data) == 60 :
            f = read_data_bio.analyze_data(data,interactions[last_index], last_index)
            for i in range(len(f) - 1) :
                cleaned_data.write(str(f[i]) + ",")
            cleaned_data.write(str(f[-1]) + "," + str(time_left) + "," + str(last_index) + "\n")
            data = []

        if (time-last_time).total_seconds() >= 1 :
            rotation = math.pow(xr**2 + yr**2 + zr**2, 1/2)
            acceleration = math.pow(xa**2 + ya**2 + za**2, 1/2)
            last_time = time
            yaw = sensor['yaw']
            pitch = sensor['pitch']
            roll = sensor['roll']
            #data.append([xr, yr, zr, xa, ya, za, heartrate, close_loc(lat, long, home_coords[0], home_coords[1])])

            data.append([rotation, acceleration, heartrate, yaw, pitch, roll, lat,long, read_data_bio.close_loc(lat, long, home_coords[0], home_coords[1])])

            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)

            read_index += 1
            print(read_index)

        if response_index == -1:
            continue
        if response_index != last_index :
            last_index = response_index
    file.close()
    cleaned_data.close()

def main() :
    out_folder = "./clean/"

    for i in range(1,6) :
        for no in [1,2] :
            if i == 1 and no == 1 :
                continue
            elif i == 2 and no == 1 :
                continue
            folder = "./dyads/H0" + str(i) + "/"
            dyad = "dyadH0" + str(i) + "A" + str(no) + "w"

            path = folder + dyad + ".prompt_groups.json"
            data_path = folder + dyad + ".sensor_data.json"
            log_path = folder + dyad + ".system_logs.log"
            responses, audio_responses = resp.read_prompts(path)
            interactions = inter.get_interactions(log_path, responses)
            freqs, home_cords = read_data_bio.find_home(data_path, 120000)

            compute_data(data_path, responses, interactions, home_cords, out_folder + dyad)

if __name__ == '__main__' :
    main()