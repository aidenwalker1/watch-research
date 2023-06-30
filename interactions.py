from datetime import datetime, timedelta

def get_interactions(log_path, responses) :
    file = open(log_path, 'r')

    st = file.readline()

    interaction_durations = [None] * len(responses)
    for i in range(len(interaction_durations)) :
        interaction_durations[i] = []

    while st :
        if "Proximity interaction ended:" in st :
            time = file.readline()
            duration = file.readline()
            time = time[time.index(' -') + 3:-1]
            duration = float(duration[duration.index('=') + 2:-2])

            if duration < 10 :
                st = file.readline()
                continue
            
            if '.' in time :
                time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
            else :
                time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            time += timedelta(hours=4)
            if time.month > 11 or (time.month == 11 and time.day >= 5) :
                time += timedelta(hours=1)
            index, time_left = in_interval(time, responses, 0)

            if index != -1 :
                interaction_durations[index].append(duration)
                

        st = file.readline()

    return interaction_durations

def compare_interactions(interactions, responses) :
    total_correct = 0
    dne = 0
    incorrects = [0,0,0]
    incorrect_time = [0,0,0]
    for i in range(len(interactions)) :
        total = sum(interactions[i])
        response = responses[i][4]
        correct = 0
        if response == -1 :
            continue
        elif response == 0 and total < 3 :
            correct = 1
        elif (response == 1 or response == 2) and total > 3:
            correct = 1

        if correct == 0 :
            incorrects[response] += 1
            incorrect_time[response] += total
            if response != 0 and total == 0 :
                dne += 1
        
        total_correct += correct
    
    return total_correct / len(interactions), incorrects, incorrect_time

def in_interval(stamp, responses, last_index=0) :
    index = last_index

    while index < len(responses) :
        response = responses[index]
        time = datetime.strptime(response[0], "%Y-%m-%dT%H:%M:%S.%f")
        diff = time - stamp
        diff = diff.total_seconds()

        if diff >= 0 :
            if diff <= 1800 :
                return index, 0
            else :
                return -1, diff
        index += 1
    return -1, 0