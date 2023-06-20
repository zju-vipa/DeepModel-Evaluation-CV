import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline
import pandas as pd
from utils import compute_pearson
# best_acc = [0.9998500347137451, 0.9997833371162415, 0.9997667074203491, 0.9998500347137451]
# model_depth = [18, 34, 50, 101]

# best_acc = [0.997316658496856, 0.999566674232482, 0.999783337116241, 0.99981665611267, 0.999850034713745, 0.99981665611267]
# sample_portion = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# plt.plot(sample_portion, best_acc)
# plt.savefig('figures/sample_portion')
# plt.close()

# fig, ax = plt.subplots()
# x = [1, 5, 10, 20, 50, 80]

# # caltech256 = [0.0259, 0.1625, 0.4563, 0.6286, 0.7554, 0.7741]
# caltech
# ax.plot(x, caltech256, label='caltech256')

# # food101 = [0.0228, 0.0429, 0.0914, 0.1733, 0.3357, 0.3824]
# ax.plot(x, food101, label='food101')

# # dtd = [0.0287, 0.1623, 0.3725, 0.5600, 0.6724, 0.6953]
# ax.plot(x, dtd, label='dtd')

# # domainnet_real = [0.0324, 0.0870, 0.2692, 0.3654, 0.5125, 0.5606]
# ax.plot(x, domainnet_real, label='domainnet_real')

# ax.set_title('transfer learning sample num')
# ax.legend()
# plt.savefig('figures/transfer_learning_sample_num')
# plt.close()
def smooth(data,weight=0.75):
    scalar = data
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


color_red_1 = (239/255, 122/255, 109/255)
color_blue_1 = (157/255, 195/255, 231/255)
color_yellow_1 = (1, 190/255, 122/255)
color_green_1 = (142/255, 207/255, 201/255)
color_grey_1 = (153/255, 153/255 ,153/255)
color_purple_1 = (190/255, 184/255, 220/255)


def visialize_transferbility_definition():

    # bmap = brewer2mpl.get_map('Set3', 'qualitative', 10)
    # colors = bmap.mpl_colors

    acc = {
        'caltech256' : [0.432, 0.4899, 0.5447, 0.5489, 0.5564, 0.5944, 0.5683, 0.6001, 0.614, 0.6114, 0.6237, 0.6191, 0.622, 0.6178, 0.6365, 0.624, 0.6305, 0.6285, 0.6256, 0.6342, 0.6117, 0.6398, 0.6274, 0.6166, 0.6109, 0.6405, 0.6432, 0.6262, 0.6032, 0.6206, 0.6314, 0.6111, 0.6513, 0.6177, 0.6231, 0.6219, 0.6418, 0.6396, 0.6489, 0.6296, 0.6415, 0.633, 0.6435, 0.6206, 0.6435, 0.6197, 0.6331, 0.6527, 0.6455, 0.6453, 0.6359, 0.6302, 0.632, 0.6467, 0.6293, 0.6428, 0.5882, 0.6441, 0.649, 0.639, 0.6506, 0.6296, 0.637, 0.6325, 0.6518, 0.6496, 0.643, 0.6513, 0.6516, 0.631, 0.6436, 0.6435, 0.6407, 0.6444, 0.6504, 0.6516, 0.6447, 0.6521, 0.6398, 0.6623, 0.6478, 0.6411, 0.6509, 0.6549, 0.6489, 0.6478, 0.6535, 0.6447, 0.6663, 0.6533, 0.6563, 0.66, 0.6597, 0.6638, 0.6381, 0.6671, 0.6572, 0.6543, 0.6677, 0.6617, 0.6567, 0.662, 0.6654, 0.6716, 0.6618, 0.6552, 0.6716, 0.6717, 0.6685, 0.6726, 0.6739, 0.6807, 0.6569, 0.6543, 0.6717, 0.6617, 0.6726, 0.6728, 0.6695, 0.6756, 0.6668, 0.6834, 0.6746, 0.6691, 0.6635, 0.6726, 0.6802, 0.6824, 0.6808, 0.6804, 0.6765, 0.6836, 0.6919, 0.6824, 0.6901, 0.6878, 0.6904, 0.6868, 0.691, 0.6938, 0.6881, 0.683, 0.6913, 0.6925, 0.6873, 0.6932, 0.6871, 0.6975, 0.693, 0.6921, 0.6896, 0.6946, 0.6981, 0.6938, 0.6953, 0.6966, 0.6949, 0.6958, 0.7023, 0.6979, 0.6904, 0.6989, 0.7009, 0.7057, 0.6984, 0.7004, 0.7063, 0.7057, 0.7046, 0.7063, 0.7043, 0.7083, 0.7117, 0.7078, 0.7114, 0.7111, 0.7091, 0.7075, 0.7078, 0.706, 0.7106, 0.699, 0.704, 0.7047, 0.7037, 0.7121, 0.706, 0.704, 0.7023, 0.7077, 0.7043, 0.7054, 0.7115, 0.7078, 0.7038, 0.7035, 0.7058, 0.6981, 0.705, 0.7091, 0.6995, 0.6984, 0.7043, 0.6969, 0.7083, 0.702, 0.7035, 0.695, 0.6983, 0.7047, 0.7017, 0.7029, 0.7026, 0.7092, 0.695, 0.7015, 0.7098, 0.7033, 0.7015, 0.7054, 0.7061, 0.7072, 0.6978, 0.7012, 0.7044, 0.702, 0.6962, 0.6996, 0.695, 0.7009, 0.6992, 0.7033, 0.7037, 0.701, 0.7072, 0.7044, 0.7003, 0.708, 0.7033, 0.7046, 0.7029, 0.7015, 0.705, 0.7037, 0.7038, 0.7118, 0.7108, 0.7055, 0.7132, 0.7029, 0.7112, 0.7086, 0.7115, 0.7132, 0.7088, 0.7098, 0.7061, 0.7084, 0.7152, 0.7118, 0.7125, 0.7143, 0.7135, 0.7165, 0.7138, 0.7152, 0.7111, 0.7064, 0.7151, 0.7115, 0.7174, 0.7121, 0.7115, 0.7155, 0.7123, 0.7149, 0.714, 0.7148, 0.7154, 0.7126, 0.7125, 0.7106, 0.7134, 0.7142, 0.7155, 0.7172, 0.7146, 0.7143, 0.7154, 0.7159, 0.7165, 0.7192, 0.716, 0.7128, 0.7165, 0.7168, 0.714, 0.7175, 0.7121, 0.7155],
        'food101' : [0.4261, 0.4883, 0.5248, 0.5651, 0.5743, 0.5806, 0.5707, 0.571, 0.5717, 0.5756, 0.6098, 0.5152, 0.5996, 0.6126, 0.6015, 0.6236, 0.6058, 0.6301, 0.6119, 0.627, 0.632, 0.6135, 0.5766, 0.5901, 0.6347, 0.5921, 0.6142, 0.6143, 0.6004, 0.6165, 0.6253, 0.6457, 0.6033, 0.5895, 0.6278, 0.6297, 0.6205, 0.6339, 0.6221, 0.6382, 0.6229, 0.6359, 0.6048, 0.6266, 0.5977, 0.6418, 0.6389, 0.6439, 0.6414, 0.6331, 0.6333, 0.6648, 0.6596, 0.64, 0.6512, 0.6706, 0.6578, 0.6373, 0.663, 0.6476, 0.6384, 0.6642, 0.6726, 0.6594, 0.6184, 0.6539, 0.6476, 0.683, 0.6598, 0.6703, 0.6549, 0.6661, 0.6671, 0.665, 0.675, 0.6835, 0.679, 0.6673, 0.6722, 0.6697, 0.6599, 0.6795, 0.6848, 0.6914, 0.688, 0.6632, 0.6758, 0.6794, 0.6555, 0.6902, 0.7036, 0.6947, 0.6819, 0.6695, 0.6932, 0.6931, 0.6891, 0.6735, 0.6947, 0.6964, 0.7022, 0.7048, 0.706, 0.6939, 0.7068, 0.6958, 0.7186, 0.6646, 0.7014, 0.6893, 0.7124, 0.7101, 0.7105, 0.6829, 0.7094, 0.7027, 0.7244, 0.7131, 0.7181, 0.7, 0.7175, 0.723, 0.722, 0.7266, 0.7184, 0.731, 0.7266, 0.7167, 0.7281, 0.7182, 0.736, 0.7279, 0.7392, 0.7378, 0.7367, 0.7283, 0.737, 0.7448, 0.7436, 0.7344, 0.743, 0.7531, 0.7383, 0.7505, 0.7465, 0.7512, 0.7452, 0.7501, 0.7486, 0.7576, 0.7503, 0.7563, 0.7577, 0.764, 0.7531, 0.7581, 0.755, 0.7617, 0.7632, 0.7588, 0.7729, 0.7632, 0.7595, 0.7626, 0.7582, 0.7691, 0.7676, 0.7748, 0.7682, 0.7718, 0.774, 0.7699, 0.7736, 0.7729, 0.7774, 0.7783, 0.7759, 0.7822, 0.7817, 0.7827, 0.7776, 0.7817, 0.7791, 0.7787, 0.7813, 0.7805, 0.7815, 0.7788, 0.7768, 0.7775, 0.7813, 0.7768, 0.772, 0.7769, 0.7792, 0.7783, 0.7735, 0.7692, 0.7793, 0.772, 0.7771, 0.773, 0.7786, 0.7761, 0.7752, 0.7775, 0.7769, 0.7731, 0.7744, 0.7785, 0.7744, 0.773, 0.7718, 0.7734, 0.7695, 0.7724, 0.7755, 0.7744, 0.7781, 0.7793, 0.7762, 0.7785, 0.7798, 0.7827, 0.7767, 0.7741, 0.777, 0.774, 0.7749, 0.773, 0.7783, 0.7762, 0.7785, 0.7766, 0.776, 0.7789, 0.7773, 0.78, 0.7771, 0.7791, 0.7842, 0.7814, 0.7821, 0.7777, 0.78, 0.78, 0.784, 0.7794, 0.7824, 0.7819, 0.781, 0.7812, 0.7808, 0.7818, 0.7874, 0.7843, 0.7843, 0.7824, 0.7815, 0.7855, 0.7869, 0.787, 0.7872, 0.7841, 0.7901, 0.7864, 0.7866, 0.7881, 0.7883, 0.7886, 0.7863, 0.7894, 0.7882, 0.7876, 0.7859, 0.7867, 0.7903, 0.7888, 0.787, 0.7891, 0.7872, 0.7894, 0.7916, 0.7905, 0.7876, 0.7924, 0.7897, 0.7943, 0.788, 0.7927, 0.7914, 0.7922, 0.7925, 0.7893, 0.7931, 0.7932, 0.7917, 0.7922, 0.7908, 0.792, 0.7943]
    }

    length_x = 100
    x = range(1, length_x+1)
    xs=np.linspace(1,length_x+1,1000)

    fig, ax = plt.subplots()
    
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    acc['caltech256'] = smooth(acc['caltech256'][:length_x])
    acc['food101'] = smooth(acc['food101'][:length_x])

    m1 = make_interp_spline(x, acc['caltech256'])
    m2 = make_interp_spline(x, acc['food101'])

    acc['caltech256'] = m1(xs)
    acc['food101'] = m2(xs)

    pos_cal = (acc['caltech256'] > acc['food101'])
    pos_food = (acc['caltech256'] <= acc['food101'])

    ax.fill_between(xs[pos_cal], acc['food101'][pos_cal], acc['caltech256'][pos_cal], facecolor=color_red_1, alpha=0.5)
    ax.fill_between(xs[pos_food], acc['food101'][pos_food], acc['caltech256'][pos_food], facecolor=color_blue_1, alpha=0.5)
    ax.plot(xs, acc['caltech256'], c=color_red_1, label='Caltech256')
    ax.plot(xs, acc['food101'], c=color_blue_1, label='Food101')

    ax.legend()
    plt.savefig('figures/transferability_definition_smooth')


def visialize_transfer_learning_epoch_num():
    file_path = {
        'caltech256' : '/nfs4/wjx/transferbility/experiment/file/file/run-tinycaltech256_2023-05-07T13_58_06.858081-tag-Test_Accuracy.csv',
        'food101' : '/nfs4/wjx/transferbility/experiment/file/file/run-tinyfood101_2023-05-07T14_00_00.756036-tag-Test_Accuracy.csv',
        'domainnet_real' : '/nfs4/wjx/transferbility/experiment/file/file/run-tinydomainnet_real_2023-05-07T14_01_22.258403-tag-Test_Accuracy.csv',
        'dtd' : '/nfs4/wjx/transferbility/experiment/file/file/run-tinydtd_2023-05-07T13_59_06.862575-tag-Test_Accuracy.csv'
    }
    acc = {}

    for name, path in file_path.items():
        data = pd.read_csv(path)
        acc[name] = data['Value'].tolist()

    length_x = 50
    x = range(1, length_x+1)
    fig, ax = plt.subplots()

    ax.plot(x, acc['caltech256'], c=color_red_1, label='Caltech256')
    ax.plot(x, acc['food101'], c=color_blue_1, label='Food101')
    ax.plot(x, acc['dtd'], c=color_yellow_1, label='DTD')
    ax.plot(x, acc['domainnet_real'], c=color_green_1, label='Domainnet_real')

    ax.legend()
    plt.savefig('figures/transfer_learning_epoch')


def visialize_pretrain_sample_num():
    file_path = {
        0.1 : '/nfs4/wjx/transferbility/experiment/file/sample_num/run-0.1cifar100_tinyfood101_resnet50_2023-05-13T08_37_42.109828-tag-Test_Accuracy.csv',
        0.2 : '/nfs4/wjx/transferbility/experiment/file/sample_num/run-0.2cifar100_tinyfood101_resnet50_2023-05-06T11_17_30.833764-tag-Test_Accuracy.csv',
        0.4 : '/nfs4/wjx/transferbility/experiment/file/sample_num/run-0.4cifar100_tinyfood101_resnet50_2023-05-13T08_40_11.243340-tag-Test_Accuracy.csv',
        0.6 : '/nfs4/wjx/transferbility/experiment/file/sample_num/run-0.6cifar100_tinyfood101_resnet50_2023-05-06T11_17_39.617049-tag-Test_Accuracy.csv',
        0.8 : '/nfs4/wjx/transferbility/experiment/file/sample_num/run-0.8cifar100_tinyfood101_resnet50_2023-05-13T10_42_46.574968-tag-Test_Accuracy.csv',
        1.0 : '/nfs4/wjx/transferbility/experiment/file/sample_num/run-1.0cifar100_tinyfood101_resnet50_2023-05-06T19_21_04.225248-tag-Test_Accuracy.csv'
    }
    acc = {}

    length_x = 50

    for name, path in file_path.items():
        data = pd.read_csv(path)
        acc[name] = data['Value'].tolist()[:length_x]

    x = range(1, length_x+1)
    fig, ax = plt.subplots()

    ax.plot(x, acc[0.1], c=color_purple_1, label='0.1')
    ax.plot(x, acc[0.2], c=color_blue_1, label='0.2')
    ax.plot(x, acc[0.4], c=color_yellow_1, label='0.4')
    ax.plot(x, acc[0.6], c=color_green_1, label='0.6')
    ax.plot(x, acc[0.8], c=color_grey_1, label='0.8')
    ax.plot(x, acc[1.0], c=color_red_1, label='1.0')


    ax.legend()
    plt.savefig('figures/pretrain_sample_num')


def visialize_transfer_learning_sample_num():
    fig, ax = plt.subplots()
    x = [1, 5, 10, 20, 50, 80]

    # caltech256 = [0.0259, 0.1625, 0.4563, 0.6286, 0.7554, 0.7741] 2epoch
    caltech256 = [0.0536, 0.4384, 0.6830, 0.7348, 0.7964, 0.8125]

    ax.plot(x, caltech256, marker='o', c=color_red_1, label='Caltech256')

    # food101 = [0.0228, 0.0429, 0.0914, 0.1733, 0.3357, 0.3824] 2epoch
    food101 = [0.0243, 0.1028, 0.2036, 0.3051, 0.4096, 0.4511]
    ax.plot(x, food101, marker='^', c=color_blue_1, label='Food101')

    # dtd = [0.0287, 0.1623, 0.3725, 0.5600, 0.6724, 0.6953]
    dtd = [0.0342, 0.2951, 0.4144, 0.4755, 0.5504, 0.5985]
    ax.plot(x, dtd, marker='p',  c=color_yellow_1, label='DTD')

    # domainnet_real = [0.0324, 0.0870, 0.2692, 0.3654, 0.5125, 0.5606]
    domainnet_real = [0.0898, 0.3935, 0.5736, 0.6518, 0.6890, 0.7177]
    ax.plot(x, domainnet_real, marker='D', c=color_green_1, label='Domainnet_real')

    relation = []
    for c, f, d, do in zip(caltech256, food101, dtd, domainnet_real):
        relation.append(compute_pearson([c, do, d, f], [4, 3, 2, 1])[0][1])
    print(relation)
    ax.legend()
    plt.savefig('figures/transfer_learning_sample_num')
    

plt.savefig('figures/transfer_learning_sample_num')
plt.close()
if __name__ == '__main__':
    # visialize_transferbility_definition()
    # visialize_transfer_learning_epoch_num()
    # visialize_transfer_learning_sample_num()
    visialize_pretrain_sample_num()