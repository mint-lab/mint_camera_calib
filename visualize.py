import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def caculate_model_score(train_error, test_error):
    test_error_weight = 1e-5
    train_error_weight = 1e-6
    ratio_weight = 1

    train_test_ratio = train_error / test_error

    return -np.log(test_error_weight / test_error + train_error_weight / train_error + ratio_weight * train_test_ratio)

def cal_aic(RMSE, dist, f, N):
    if dist == 'no':
        num_para = len(f.split('_'))
    else:
        num_para = len(f.split('_')) + len(dist.split('_'))

    return 2 * num_para + N * np.log(pow(RMSE, 2))

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    array = (array - min_val) / (max_val - min_val)

    return array

if __name__ == '__main__':
    
    file_path = 'data_synthetic/fisheye_40/results/' 
    # train_error = np.load(file_path + 'train.npy')
    # test_error = np.load(file_path + 'test.npy')
    # score = caculate_model_score(train_error, test_error)
    # score = normalize(score)
    # score = np.load(file_path + 'criteria_proposal.npy')
    AIC = np.load(file_path + '/AIC.npy')
    AIC = normalize(AIC)

    # AIC_confusion_matrix = max_value - AIC_confusion_matrix
    index = ['PN1', 'PN2', 'PN3', 'PN4']
    column = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'KB1', 'KB2', 'KB3', 'KB4', 'KB5']
    index1= ['PF1', 'PF2', 'PF3', 'PF4']
    
    df = pd.DataFrame(AIC, index=index, columns=column)
    
    
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=0.65) # for label size
    sn.heatmap(df, annot=True, cmap='jet', square=True) # font size
    plt.plot([5, 5], [0, 4], 'k-', linewidth=3)
    # plt.axis('equal')
    plt.show()
    