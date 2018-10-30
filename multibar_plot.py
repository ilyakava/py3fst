import numpy as np
import matplotlib.pyplot as plt

import pdb

# median
noisy2 = np.array([4.46, -1.09, 5.41, -6.06, -7.68])
class2 = np.array([4.96, 1.84, 0.52, -0.0, -0.0])
pil2 = np.array([7.88, 3.56, 5.95, -1.22, 0.86])
noisy3 = np.array([-0.25, -4.86, -0.23, -9.28, -11.29])
class3= np.array([2.31, -0.96, -0.0, -2.84, -1.69])
pil3 = np.array([3.33, -0.66, 0.84, -8.87, -1.64])
noisy5 = np.array([-12.24, -12.53, -1.73, -5.16, -13.63])
class5 = np.array([-10.23, -12.79, 0.12, 0.28, -2.01])



N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

mylim = 255.0
mygray = (171/mylim,171/mylim,171/mylim)
mygreen = (66/mylim,142/mylim,99/mylim)
myyellow = (239/mylim,200/mylim,88/mylim)
myred = (176/mylim,69/mylim,65/mylim)
myblue = (121/mylim, 160/mylim, 240/mylim)

barhs1 = np.append(class2-noisy2, np.array([(class2-noisy2).mean()]))
rects1 = ax.bar(ind, barhs1, width, color=myblue)

barhs2 = np.append(pil2-noisy2, np.array([(pil2-noisy2).mean()]))
rects2 = ax.bar(ind+width, barhs2, width, color=myred)

barhs3 = np.append(class3-noisy3, np.array([(class3-noisy3).mean()]))
rects3 = ax.bar(ind+width*2, barhs3, width, color=myyellow)

barhs4 = np.append(pil3-noisy3, np.array([(pil3-noisy3).mean()]))
rects4 = ax.bar(ind+width*3, barhs4, width, color=mygreen)

barhs5 = np.append(class5-noisy5, np.array([(class5-noisy5).mean()]))
rects5 = ax.bar(ind+width*4, barhs5, width, color=mygray)

ax.set_title('Median Scale Invariant SDR Improvement')
ax.set_ylabel('dB')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Acc', 'Pass', 'Idle', 'Horn', 'Steps', 'Avg') )
ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
	('Class-2', 'PIL-2', 'Class-3', 'PIL-3', 'Class-5') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)
ax.grid()
plt.savefig('/cfarhomes/ilyak/Desktop/icassp_plot.png')
# plt.savefig('/Users/artsyinc/Desktop/icassp_plot.png')
