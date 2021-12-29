import fire
import pandas
from tqdm import trange
import csv
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt

def loadcsv(path, line=1063):
    #read the csv file and treat the data as array
    arr_list = []
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            arr_list.append(row)

    data_mat = np.zeros((len(arr_list)-1, len(arr_list[1])-2))
    for k in range(1, len(arr_list)):
        data_mat[k-1] = np.asarray(arr_list[k][2:], np.float32)
    
    br = np.mean(data_mat[:,2])
    bp = np.mean(data_mat[:,3])
    asa = np.mean(data_mat[:,6])
    co = np.mean(data_mat[:,9])
    n_sp = np.mean(data_mat[:,15])

    stat = [n_sp, asa, br, bp, co]
    return np.asarray(stat)


def main(path='./collect_eval_res/'):
    our1l_res_path = path
    if '/NYU' in path:
        num_list = [300, 432, 588, 768, 972, 1200, 1452, 1728, 2028, 2352]
    else:
        num_list = [54, 96, 150, 216, 294, 384, 486, 600, 726, 864, 1014, 1176]
    n_set = len(num_list)
    Ours = np.zeros((n_set, 5))
    for i in trange(n_set):
        load_path = os.path.join(our1l_res_path + f'/SPixelNet_nSpixel_{num_list[i]}/map_csv/results.csv')
        Ours[i] = loadcsv(load_path)
    leg_font = {'family': 'Times New Roman', 'size': 16}
    label_font = {'family': 'Times New Roman', 'size':16}
    #plot ASA
    plt.figure('ASA')
    asa_d = plt.plot(Ours[:,0], Ours[:, 1], 'k-^', markerfacecolor='k', markersize=6, label='SSN-FCN')
    plt.legend([asa_d], labels=['SSN-FCN'], loc='lower right', prop=leg_font)
    
    x_min, x_max = np.min(Ours[:,0]), np.max(Ours[:,0])
    y_min, y_max = np.min(Ours[:,1]), np.max(Ours[:,1])
    x_min, x_max = x_min - 10, x_max + 10
    y_min, y_max = y_min - 0.005, y_max + 0.005

    #x_end = np.max(Ours[:,0]) + 10
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

    plt.xlabel('Number of Superpixels', label_font)
    plt.ylabel('ASA Score', label_font)
    plt.savefig(os.path.join(our1l_res_path, 'ASA.jpg'))

    #plot CO
    plt.figure('CO')
    co_d = plt.plot(Ours[:,0], Ours[:, 4], 'k-^', markerfacecolor='k', markersize=6, label='SSN-FCN')
    plt.legend([co_d], labels=['SSN-FCN'], loc='lower right', prop=leg_font)

    x_min, x_max = np.min(Ours[:,0]), np.max(Ours[:,0])
    y_min, y_max = np.min(Ours[:,4]), np.max(Ours[:,4])
    x_min, x_max = x_min - 10, x_max + 10
    y_min, y_max = y_min - 0.01, y_max + 0.01

    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

    plt.xlabel('Number of Superpixels', label_font)
    plt.ylabel('CO Score', label_font)
    plt.savefig(os.path.join(our1l_res_path, 'CO.jpg'))

    #plot BR-BP 
    plt.figure('BR-BP')
    br_bp_d = plt.plot(Ours[:,2], Ours[:, 3], 'k-^', markerfacecolor='k', markersize=6, label='SSN-FCN')
    plt.legend([br_bp_d], labels=['SSN-FCN'], loc='lower right', prop=leg_font)
    x_min, x_max = np.min(Ours[:,2]), np.max(Ours[:,2])
    y_min, y_max = np.min(Ours[:,3]), np.max(Ours[:,3])
    x_min, x_max = x_min - 0.01, x_max + 0.01
    y_min, y_max = y_min - 0.01, y_max + 0.01
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))

    plt.xticks(np.arange(x_min, x_max, 0.05).tolist())
    plt.yticks(np.arange(y_min, y_max, 0.01).tolist())
    plt.xlabel('Boundary Recall', label_font)
    plt.ylabel('Boundary Precision', label_font)
    plt.savefig(os.path.join(our1l_res_path, 'BR-BP.jpg'))

    overall_mean = np.mean(Ours, axis=0, keepdims=True)
    results_map = np.concatenate([Ours, overall_mean], axis=0)
    np.savetxt(os.path.join(path, 'mean_result.txt'), results_map, fmt='%.05f')

fire.Fire(main)
