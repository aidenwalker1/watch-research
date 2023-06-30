import interactions as inter
from datetime import datetime
import json
import numpy as np

#reads watch json data and converts it into clean and usable data
def read_sensor_data_responses(path, responses, read_range = 10000) :
    rssi_current = 0

    bio_file = open(path, 'r')

    #output list
    all_data = [0] * len(responses)
    cur = []
    
    #data over given seconds - fills with rssi value of -85
    rssi_avg_minute = [-85] * 20
    rate_of_change = [-85] * 3
    deviations = [-85] * 5

    rssi_avg = 0
    stddev = 0
    
    #current time in rssi
    (time_60, time_70, time_80) = (0,0,0)

    st = ''
    fields = None

    interaction = 0
    last_response = 0

    #goes through file
    for i in range(0,read_range) :
        #reads data, fills dictionary with json value
        st = bio_file.readline()
        try :
            fields = json.loads(st[:st.find(',\n')])
        except:
            break

        stamp = str(fields["stamp"])  
        stamp = stamp[:stamp.index("+")]

        time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")

        response_index, time_left = inter.in_interval(time, responses, last_response)
        sensor = fields["sensors"]

        if response_index != last_response :
            all_data[last_response] = cur
            cur = []
            last_response = response_index

        rssi_current = sensor["peer_bt_rssi"] 

        #fills data arrays with current rssi
        rssi_avg_minute[i % 20] = rssi_current
        deviations[i % 5] = rssi_current
        rate_of_change[i % 3] = rssi_current

        #calculates average, rate of change, standard deviation
        rssi_avg = np.average(rssi_avg_minute)

        #update times based on current rssi
        (time_60, time_70, time_80) = update_times(rssi_avg, (time_60, time_70, time_80))

        stddev = np.std(deviations)
        
        #fills attribute list with fields
        cur.append([rssi_avg, stddev, time_60, time_70, time_80])
    bio_file.close()
    return all_data

def read_sensor_data_times(path, times, read_range = 10000) :
    rssi_current = 0

    bio_file = open(path, 'r')
    bio_file.readline()

    #output list
    bio_test = []
    
    #data over given seconds - fills with rssi value of -85
    rssi_avg_minute = [-85] * 20
    rate_of_change = [-85] * 3
    deviations = [-85] * 5

    num_features = 5

    attribute_list = [0] * (num_features + 1)

    rssi_avg = 0
    stddev = 0
    
    #current time in rssi
    (time_60, time_70, time_80) = (0,0,0)

    st = ''
    fields = None

    interaction = 0
    i = 0
    #goes through file
    while i < read_range:
        #reads data, fills dictionary with json value
        st = bio_file.readline()
        try:
            fields = json.loads(st[:st.find(',\n')])
        except:
            break
        
        if fields['sensors'].get('peer_bt_rssi') == None :
            continue

        sensor = fields["sensors"] 
        rssi_current = sensor['peer_bt_rssi']

        #fills data arrays with current rssi
        rssi_avg_minute[i % 20] = rssi_current
        deviations[i % 5] = rssi_current
        rate_of_change[i % 3] = rssi_current

        #calculates average, rate of change, standard deviation
        rssi_avg = np.average(rssi_avg_minute)

        #update times based on current rssi
        (time_60, time_70, time_80) = update_times(rssi_avg, (time_60, time_70, time_80))

        stddev = np.std(deviations)

        #checks if interaction is happening in current time
        interaction = interaction_happened(times, i)
        
        #fills attribute list with fields
        fill_attributes(attribute_list, rssi_avg, stddev, time_60, time_70, time_80, interaction)

        #saves current attribute list
        bio_test.append(attribute_list.copy())
        i += 1
    bio_file.close()
    return bio_test


#updates the times given the current rssi
def update_times(rssi_current, times) :
    time_60, time_70, time_80 = times
    if rssi_current >= -60 :
        time_60 = time_60 + 1
        time_70 = time_70 + 1
    elif rssi_current >= -70 :
        time_70 = time_70 + 1
        time_80 = time_80 + 1
    elif rssi_current >= -80 :
        time_80 = time_80 + 1
        time_60 = time_60 - 1 if time_60 > 0 else 0
    else :
        if rssi_current >= -85 :
            time_80 = time_80 - 1 if time_80 > 0 else 0
            time_70 = time_70 - 0.5 if time_70 > 0 else 0
            time_60 = 0
        else :            
            time_60 = 0   
            time_80 = 0
            time_70 = 0
    
    return (time_60, time_70, time_80)

#fills list with given attributes
def fill_attributes(attribute_list,rssi_avg, stddev, time_60, time_70, time_80, interaction) :
    attribute_list[0] = rssi_avg
    attribute_list[1] = stddev
    attribute_list[2] = time_60
    attribute_list[3] = time_70
    attribute_list[4] = time_80

    attribute_list[5] = interaction

#detects if current time is in given interaction times
def interaction_happened(times, time) :
    for cur_time in times: 
        if cur_time[0] <= time <= cur_time[1] :
            return 1

    return 0
