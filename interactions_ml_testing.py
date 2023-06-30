import random

import numpy as np

import matplotlib.pyplot as plt
import interactions as inter
import responses as resp
import read_data_bt
import interaction_model

num_features = 5

# plots average rssi
def plot_data(data, lower, upper) :
    y = [data[i][0] for i in range(lower,upper)]
    baseline = [-80 for i in range(lower,upper)]

    x = [i for i in range(lower,upper)]
    fig, ax = plt.subplots()
    ax.plot(x, y, "red")
    ax.plot(x, baseline, "b--")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Bluetooth RSSI distance")
    plt.title("Distance of 2 watch users")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 300))
    plt.show()

# plots model vs actual interaction data
def plot_model(X,y, model) :
    x = [i for i in range(len(y))]
    predictions = model.predict(X) 
    correct_outputs = [y[i] for i in range (len(y))]

    fig, ax = plt.subplots()
    ax.plot(x, predictions, "r--")
    ax.plot(x, correct_outputs, "b")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Interaction occured")
    plt.title("ML Model vs Actual Data")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 300))
    plt.show()

# main function
def run_program() :
    random.seed(None)
    
    already_trained = True

    # interaction times
    times = [
        (150,515),
        (1285,1575),
        (1987,2010),
        (2315,2385),
        (2930,3210),
        (3500,3905),
        (4340,4600),
        (5090,5287)
    ]

    # file read data
    path = "./sensor_data/sensor_data.json"
    read_range = 6000

    # gets models -> base model is used to help mlp model which is the main one used
    #base_model, mlp = form_models(path, read_range, times, already_trained)
    
    #gets test data : for now 0-3000 is data used for training and 3000-6000 is fresh test data
    file2_range = 3600
    file2_times = [
        (45,65),
        (270,290),
        (435,465),
        (710,727),
        (925,1295),
        (1590,1617),
        (2020,2050),
        (2415,2685),
        (3010,3350),
        (3515,3565)
    ]

    file2_path = "./sensor_data/sensor_data1.json"

    data1 = read_data_bt.read_sensor_data_times(path, times, read_range)

    data2 = read_data_bt.read_sensor_data_times(file2_path, file2_times, file2_range)

    data1 = np.array(data1)
    data2 = np.array(data2)

    data = np.append(data1,data2,axis=0)
    base_mod, mlp = interaction_model.form_models(data, already_trained)

    p = './dyadH01A1w_clean_bt.json'

    responses, audio_responses = resp.read_prompts('./dyads/H01/dyadH01A1w.prompt_groups.json')
    td = read_data_bt.read_sensor_data_responses(p, responses, 500000)
    sums = []

    for t in td :
        if t == 0 :
            sums.append([])
            continue
        d = interaction_model.append_preds(t, base_mod)
        predictions = mlp.predict(d)
        sums.append(predictions)

    ratio, incorrects, incorrect_times = inter.compare_interactions(sums, responses)
    # testdata = np.array(read_data(p, test_times, 50000, base_mod))

    # X = testdata[:,:num_features+1]
    # y = testdata[:,num_features+1]
    # predictions = mlp.predict(X) 
    
    # # tests model and plots data
    # interaction_model.test_accuracy(mlp, X, y)

    # plot_model(X,y, mlp)
    # plot_data(new_test_arr, 0, 6600)
    # plot_data(data1, 0, 6000)

if __name__ == '__main__' :
    run_program()