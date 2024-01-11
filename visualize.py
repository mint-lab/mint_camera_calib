import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def caculate_model_score(train_error, test_error):
    test_error_weight = 1
    train_error_weight = 1
    ratio_weight = 1

    train_test_ratio = train_error / test_error

    # return -np.log(test_error_weight / test_error + train_error_weight / train_error + ratio_weight * train_test_ratio)
    num_para = get_num_para()
    return np.log(train_error + test_error + pow(train_error, 2) / test_error + num_para * train_error)

def get_num_para():
    dists = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4]
    intrinsics = [1, 2, 3, 4]

    num_para = []
    for intrinsic in intrinsics:
        tmp = []
        for dist in dists:
            tmp.append(dist + intrinsic)
        num_para.append(tmp)

    return np.array(num_para)

def cal_AIC(RMSE, N_samples):
    proj_model_num_para = [1, 2, 3, 4]
    dist_model_num_para = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4]

    num_para = np.array([[dist + f for dist in dist_model_num_para] for f in proj_model_num_para])

    return N_samples * np.log(pow(RMSE, 2)) + 2 * num_para

def cal_BIC(RMSE, N_samples):
    proj_model_num_para = [1, 2, 3, 4]
    dist_model_num_para = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4]

    num_para = np.array([[dist + f for dist in dist_model_num_para] for f in proj_model_num_para])

    return N_samples * np.log(pow(RMSE, 2)) + num_para * np.log10(N_samples)

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    array = (array - min_val) / (max_val - min_val)

    return array

def load_score(file_path):
    AIC_score = np.load(file_path + 'AIC/AIC.npy')
    AIC_score = normalize(AIC_score)

    BIC_score = np.load(file_path + 'AIC/BIC.npy')
    BIC_score = normalize(BIC_score)

    train_error = np.load(file_path + 'proposal/train.npy')
    test_error = np.load(file_path + 'proposal/test.npy')
    proposal_score = caculate_model_score(train_error, test_error)
    proposal_score = normalize(proposal_score)

    return AIC_score, BIC_score, proposal_score

def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    array = (array - min_val) / (max_val - min_val)

    return array

def plot_one_score(df):
    
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=0.65) # for label size
    sn.heatmap(df, annot=True, cmap='jet', square=True) # font size
    plt.plot([5, 5], [0, 4], 'k-', linewidth=3)
    # plt.axis('equal')
    plt.show()

def plot_multi_score(df_AIC, df_BIC, df_proposal):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7))
    sn.set(font_scale=0.7)

    # Plot the first heatmap
    AIC = sn.heatmap(df_AIC, ax=axes[0], annot=True, cmap='jet', square=True)
    AIC.set_xticklabels(AIC.get_xticklabels(), fontsize=8)
    AIC.set_yticklabels(AIC.get_yticklabels(), fontsize=8)
    axes[0].plot([5, 5], [0, 4], 'k-', linewidth=2.5)
    axes[0].set_title('AIC', fontsize=12)

    # Plot the second heatmap
    BIC = sn.heatmap(df_BIC, ax=axes[1],  annot=True, cmap='jet', square=True)
    BIC.set_xticklabels(BIC.get_xticklabels(), fontsize=8)
    BIC.set_yticklabels(BIC.get_yticklabels(), fontsize=8)
    axes[1].plot([5, 5], [0, 4], 'k-', linewidth=2.5)
    axes[1].set_title('BIC', fontsize=12)

    # Plot the third heatmap
    proposal = sn.heatmap(df_proposal, ax=axes[2],  annot=True, cmap='jet', square=True)
    proposal.set_xticklabels(proposal.get_xticklabels(), fontsize=8)
    proposal.set_yticklabels(proposal.get_yticklabels(), fontsize=8)
    axes[2].plot([5, 5], [0, 4], 'k-', linewidth=2.5)
    axes[2].set_title('Proposal', fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_path = 'data/synthetic/dataset_noise_1/model_23' 
    index = ['P0', 'P1', 'P2', 'P3']
    column = ['BC0', 'BC1', 'BC2', 'BC3', 'BC4', 'KB0', 'KB1', 'KB2', 'KB3', 'KB4']
    N_samples = 40 * 70

    # Load error
    rms = np.load(os.path.join(file_path, 'rms.npy'))
    train_error = np.load(os.path.join(file_path, 'train_error_extrapolar.npy'))
    test_error = np.load(os.path.join(file_path, 'test_error_extrapolar.npy'))

    # AIC 
    AIC = cal_AIC(rms, N_samples)
    AIC = normalize(AIC)
    df_AIC = pd.DataFrame(AIC, index=index, columns=column)
    # plot_one_score(df_AIC)

    # BIC
    BIC = cal_BIC(rms, N_samples)
    BIC = normalize(BIC)
    df_BIC = pd.DataFrame(BIC, index=index, columns=column)

    # Proposal
    proposal_core = caculate_model_score(train_error, test_error)
    proposal_core = normalize(proposal_core)
    df_proposal = pd.DataFrame(proposal_core, index=index, columns=column)

    # Plot AIC, BIC, proposal
    plot_multi_score(df_AIC, df_BIC, df_proposal)