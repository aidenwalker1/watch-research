import json

def read_prompts(path) :
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    output = []
    audio_output = []
    
    for d in data :
        prompts = d['prompts']
        arr = [-1] * 9
        save_stamp = None
        for i in range(len(prompts)) :
            prompt = prompts[i]

            index = prompt['prompt_name'][-1]
            type = prompt['prompt_type']

            if not index.isnumeric():
                if type == 'activity_audio_log' :
                    audio_output.append((stamp, value))
                continue

            index = int(index)
            stamp = prompt['response_stamp']
            
            value = prompt['chosen_response']
            if value != None :
                value = value['response_value']

                if index == 4:
                    value = time_to_val(value)
            else :
                value = -1

            if stamp == None :
                value = -1
            else :
                stamp = stamp[:stamp.index("+")]

                if save_stamp == None :
                    save_stamp = stamp
                    
            arr[index] = int(value)
                
        if save_stamp != None :
            arr[0] = save_stamp
            output.append(arr)
    
    return output, audio_output

def time_to_val(time) :
    if time == 'None' :
        return 0
    elif 'Less' in time :
        return 1
    else :
        return 2