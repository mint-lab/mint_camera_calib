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

def load_resuls(criteria_type, file_path):
    if criteria_type == 'aic':
        score = np.load(file_path + 'AIC.npy')
    elif criteria_type == 'bic':
        score = np.load(file_path + 'BIC.npy')
    else:
        train_error = np.load(file_path + 'train.npy')
        test_error = np.load(file_path + 'test.npy')
        score = caculate_model_score(train_error, test_error)
    score = normalize(score)

    return score


if __name__ == '__main__':
    file_path = 'results/synthetic/normal_40_random/' 
    score =  load_resuls('bic', file_path)

    # AIC_confusion_matrix = max_value - AIC_confusion_matrix
    index = ['PN0', 'PN1', 'PN2', 'PN3']
    column = ['BC0', 'BC1', 'BC2', 'BC3', 'BC4', 'KB0', 'KB1', 'KB2', 'KB3', 'KB4']
    index1= ['PF0', 'PF1', 'PF2', 'PF3']
    df = pd.DataFrame(score, index=index, columns=column)
    
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=0.65) # for label size
    sn.heatmap(df, annot=True, cmap='jet', square=True) # font size
    plt.plot([5, 5], [0, 4], 'k-', linewidth=3)
    # plt.axis('equal')
    plt.show()
    