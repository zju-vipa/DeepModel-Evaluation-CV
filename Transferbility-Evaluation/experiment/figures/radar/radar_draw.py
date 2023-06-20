import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import math

color_red_1 = (239/255, 122/255, 109/255)
color_blue_1 = (157/255, 195/255, 231/255)
color_yellow_1 = (1, 190/255, 122/255)
color_green_1 = (142/255, 207/255, 201/255)
color_grey_1 = (153/255, 153/255 ,153/255)
color_purple_1 = (190/255, 184/255, 220/255)

def norm(x):
    x = 200 * math.atan(x)
    x /= math.pi
    if x <= 0:
        x += 100
    return x

categories = [
    '输入', 
    '输出特征', 
    '输出概率'
]

# values = {
#     'vgg' : [3.509, -1.283, -0.628],
#     'resnet' : [3.509, -0.508, -0.548],
#     'mobilenet' : [3.509, -8.905, -1.492],
#     'densenet' :[3.509, -0.703, -0.481]
# }

#infograph
# values = {
#     'vgg' : [4.161, -14.717, -1.59],
#     'resnet' : [4.161, -16.025, -1.829],
#     'mobilenet' : [4.161, -22.183, -2.114],
#     'densenet' :[4.161, -15.742, -1.785]
# }

# caltech256
values = {
    'vgg' : [3.352, -1.480, -0.719],
    'resnet' : [3.352,  -0.960,  -0.646],
    'mobilenet' : [3.352, -18.283,  -1.736],
    'densenet' :[3.352,  -0.594,  -0.542]
}

# dtd
# values = {
#     'vgg' : [3.146, -37.972 , -1.950],
#     'resnet' : [3.146,  -36.400,   -1.878],
#     'mobilenet' : [3.146,  -61.611,  -2.767],
#     'densenet' :[3.146,   -46.203,  -1.609]
# }

# densenet
# values = {
#     'food101' : [2.569, -119.706, -2.357],
#     'caltech256' : [3.352, -0.594, -0.542],
#     'dtd' : [3.146, -46.203, -1.609],
#     'real' : [3.509, -0.703, -0.481]
# }

colos = {
    'vgg' : color_blue_1,
    'resnet' : color_red_1,
    'mobilenet' : color_green_1,
    'densenet' : color_purple_1
}

# colos = {
#     'food101' : color_blue_1,
#     'caltech256' : color_red_1,
#     'dtd' : color_green_1,
#     'real' : color_purple_1
# }

my_font = FontProperties(fname=r"fonts/SimSun.ttc", size=17, weight='heavy')

for name, value in values.items():
    for i in range(0, len(value)):
        value[i] = norm(value[i])


angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
grid_rs = np.linspace(0, 100, 5, endpoint=False)
new_angles = np.concatenate((angles, [angles[0]]))

for name, value in values.items():
    values[name] = np.concatenate((value, [value[0]]))


fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

# for name, value in values.items():
#     print(value[0])
#     ax.plot(new_angles, value, 'o-', markersize= 1, color=colos[name], linewidth=1)
#     ax.fill(new_angles, value, facecolor=colos[name], alpha=0.2)

name = 'densenet'
ax.plot(new_angles, values[name], 'o-', color=colos[name], linewidth=2)
ax.fill(new_angles, values[name], facecolor=colos[name], alpha=0.2)

ax.set_rlim(0, 100)
ax.grid(True)
lines, labels = ax.set_thetagrids(angles * 180 / np.pi, categories, fontproperties=my_font)
for label in labels:
    p = label.get_position()
    label.set_position((p[0], -0.1))

plt.savefig(f'caltech256_{name}(c).png')