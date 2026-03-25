#%%
import  numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#%%
# baseline: Packet id: 8094 YZGlobalFlit_id 74554 YZGlobalFlitPass 202566 YZGlobalRespFlitPass 158610 yzWeightCollsionInRouterCountSum 5849232 yzWeightCollsionInNICountSum 74554 yzFlitCollsionCountSum 202566 tempRouterNetWholeFlipCount 21178213 tempRouterNetWholeFlipCount_fix35 5849232
#half YZGlobalFlit_id 74554 YZGlobalFlitPass 202566 YZGlobalRespFlitPass 158610 yzWeightCollsionInRouterCountSum 5373406 yzWeightCollsionInNICountSum 74554 yzFlitCollsionCountSum 202566 tempRouterNetWholeFlipCount 18525122 tempRouterNetWholeFlipCount_fix35 5373406
lenet_float_results = np.array([ 21178213, 18525122 ,16199367 ])
lenet_fixed_results  = np.array([ 5849232, 5373406, 4857909])
#lenet_float_results = np.array([ 21771412, 18218297 ,16789674])
#lenet_fixed_results  = np.array([ 5616286, 4889065, 4693078])
# Packet id: 8094 YZGlobalFlit_id 74554 YZGlobalFlitPass 202566 YZGlobalRespFlitPass 158610 yzWeightCollsionInRouterCountSum 5840312 yzWeightCollsionInNICountSum 74554 yzFlitCollsionCountSum 202566 tempRouterNetWholeFlipCount 21458990 tempRouterNetWholeFlipCount_fix35 5840312
#Packet id: 8094 YZGlobalFlit_id 74554 YZGlobalFlitPass 202566 YZGlobalRespFlitPass 158610 yzWeightCollsionInRouterCountSum 3741965 yzWeightCollsionInNICountSum 74554 yzFlitCollsionCountSum 202566 tempRouterNetWholeFlipCount 14590875 tempRouterNetWholeFlipCount_fix35 3741965
lenet_float_resultsRealWeight = np.array([ 21458990,  17473331 ,14590875])
lenet_fixed_resultsRealWeight  = np.array([ 5840312, 4803903,  3741965])


#
# baseline Cycles: 2007154 Packet id: 439136 YZGlobalFlit_id 2683216 YZGlobalFlitPass 7283045 YZGlobalRespFlitPass 4899153 yzWeightCollsionInRouterCountSum 4599829 yzWeightCollsionInNICountSum 2683216 yzFlitCollsionCountSum 7283045 tempRouterNetWholeFlipCount 627890641 tempRouterNetWholeFlipCount_fix35 172494247 !!END!! 运行时间: 206.454 秒
# half order this is the last layer Cycles: 2007154 Packet id: 439136 YZGlobalFlit_id 2683216 YZGlobalFlitPass 7283045 YZGlobalRespFlitPass 4899153 yzWeightCollsionInRouterCountSum 4599829 yzWeightCollsionInNICountSum 2683216 yzFlitCollsionCountSum 7283045 tempRouterNetWholeFlipCount 493926234 tempRouterNetWholeFlipCount_fix35 131426276 !!END!! 运行时间: 218.838 秒
# allordered: Cycles: 2007154 Packet id: 439136 YZGlobalFlit_id 2683216 YZGlobalFlitPass 7283045 YZGlobalRespFlitPass 4899153 yzWeightCollsionInRouterCountSum 4599829 yzWeightCollsionInNICountSum 2683216 yzFlitCollsionCountSum 7283045 tempRouterNetWholeFlipCount 438820706 tempRouterNetWholeFlipCount_fix35 124826079 !!END!! 运行时间: 228.374 秒



darknet64_float_results  =np.array([13739180266,11861466407,9906869199])
darknet64_fixed_results = np.array([4193610478, 3909065297, 2480505268])

vgg_float_results  =  np.array([627890641, 493926234 , 438820706])
vgg_fixed_results = np.array([ 172494247, 131426276, 124826079])
#%%# Normalize each group to its baseline
# Normalize each group's bars to the baseline
lenet_float_normalized = lenet_float_results / lenet_float_results[0] * 100
lenet_fixed_normalized = lenet_fixed_results / lenet_fixed_results[0] * 100
vgg_float_normalized = vgg_float_results / vgg_float_results[0] * 100
vgg_fixed_normalized = vgg_fixed_results / vgg_fixed_results[0] * 100
lenet_floatRealWeight_normalized = lenet_float_resultsRealWeight / lenet_float_resultsRealWeight[0] * 100
lenet_fixedRealWeight_normalized = lenet_fixed_resultsRealWeight / lenet_fixed_resultsRealWeight[0] * 100
darknet64_float_normalized  = darknet64_float_results / darknet64_float_results[0] * 100
darknet64_fixed_normalized = darknet64_fixed_results / darknet64_fixed_results[0] * 100
# Plotting setup
group_width = 0.8
bar_width = group_width / 4
index = np.arange(3)  # 3 bars per group
x = np.arange(18)  # Total number of bars

fig, ax = plt.subplots(figsize=(12, 8))

# Plotting the bars
bars1 = ax.bar(x[0:3], lenet_float_normalized, bar_width, label='LeNet float-32 random')#, color='b'
bars2 = ax.bar(x[3:6], lenet_fixed_normalized, bar_width, label='LeNet fixed-8 random')#, color='g'
bars3 = ax.bar(x[6:9], lenet_floatRealWeight_normalized, bar_width, label='LeNet float-32 trained')#, color='m'
bars4 = ax.bar(x[9:12], lenet_fixedRealWeight_normalized, bar_width, label='LeNet fixed-8 trained')#
#bars5 = ax.bar(x[12:15], vgg_float_normalized, bar_width, label='VGG Float', color='r')
#bars6 = ax.bar(x[15:18], vgg_fixed_normalized, bar_width, label='VGG Fixed', color='c')
bars5 = ax.bar(x[12:15], darknet64_float_normalized, bar_width, label='DarkNet float-32 random')# , color='r'
bars6 = ax.bar(x[15:18], darknet64_fixed_normalized, bar_width, label='DarkNet fix-8 random')# , color='c'
# Adding text for labels, title, and axes ticks
ax.set_xlabel('Configurations')
ax.set_ylabel('Bit transition (%)')
#ax.set_title('Performance Comparison Across Different Models and Configurations')
ax.set_xticks(x + bar_width / 2)
#ax.set_xticklabels([' Baseline', ' HalfOrdered', 'AllOrdered', ' Baseline', 'HalfOrdered', 'AllOrdered',
#                    ' Baseline', ' HalfOrdered', 'AllOrdered', 'Baseline', 'HalfOrdered', 'AllOrdered', 'Baseline', 'HalfOrdered', 'AllOrdered', 'Baseline', 'HalfOrdered', 'AllOrdered'])
ax.set_xticklabels([])
# Adding labels to the baseline bars
ax.text(bars1[0].get_x() + bars1[0].get_width() / 2, bars1[0].get_height(), f'{lenet_float_results[0]:,.0f}', ha='center', va='bottom')
ax.text(bars2[0].get_x() + bars2[0].get_width() / 2, bars2[0].get_height(), f'{lenet_fixed_results[0]:,.0f}', ha='center', va='bottom')
ax.text(bars3[0].get_x() + bars3[0].get_width() / 2, bars3[0].get_height(), f'{lenet_float_resultsRealWeight[0]:,.0f}', ha='center', va='bottom')
ax.text(bars4[0].get_x() + bars4[0].get_width() / 2, bars4[0].get_height(), f'{lenet_fixed_resultsRealWeight[0]:,.0f}', ha='center', va='bottom')

#ax.text(bars5[0].get_x() + bars5[0].get_width() / 2, bars5[0].get_height(), f'{vgg_float_results[0]:,.0f}', ha='center', va='bottom')
#ax.text(bars6[0].get_x() + bars6[0].get_width() / 2, bars6[0].get_height(), f'{vgg_fixed_results[0]:,.0f}', ha='center', va='bottom')
ax.text(bars5[0].get_x() + bars5[0].get_width() / 2, bars5[0].get_height(), f'{darknet64_float_results[0]:,.0f}', ha='center', va='bottom')
ax.text(bars6[0].get_x() + bars6[0].get_width() / 2, bars6[0].get_height(), f'{darknet64_fixed_results[0]:,.0f}', ha='center', va='bottom')
#之前的图没错，但是text浮动文字有问题，搞成了vgg的，也就是显示了一个无关的值。
# Adding a legend
ax.legend()
plt.tight_layout()  # 自动调整以减少多余的空白
plt.show()
#%%
# different lenet layers
# layer 1 :

# Calculating the ratio of [2] to [0] for each array
ratios = {
    "lenet_float_ratio": lenet_float_results[1] / lenet_float_results[0],
    "lenet_fixed_ratio": lenet_fixed_results[1] / lenet_fixed_results[0],
    "lenet_float_realWeight_ratio": lenet_float_resultsRealWeight[1] / lenet_float_resultsRealWeight[0],
    "lenet_fixed_realWeight_ratio": lenet_fixed_resultsRealWeight[1] / lenet_fixed_resultsRealWeight[0],
    "MC8_8x8lenet_float_ratio": MC8_8x8lenet_float_results[1] / MC8_8x8lenet_float_results[0],
    "MC8_8x8lenet_fixed_ratio": MC8_8x8lenet_fixed_results[1] / MC8_8x8lenet_fixed_results[0],
    "MC8_8x8lenet_float_realWeight_ratio": MC8_8x8lenet_float_realWeights[1] / MC8_8x8lenet_float_realWeights[0],
    "MC8_8x8lenet_fixed_realWeight_ratio": MC8_8x8lenet_fixed_realWeights[1] / MC8_8x8lenet_fixed_realWeights[0],
    "MC4_8x8lenet_float_ratio": MC4_8x8lenet_float_results[1] / MC4_8x8lenet_float_results[0],
    "MC4_8x8lenet_fixed_ratio": MC4_8x8lenet_fixed_results[1] / MC4_8x8lenet_fixed_results[0],
    "MC4_8x8lenet_float_realWeight_ratio": MC4_8x8lenet_float_realWeights[1] / MC4_8x8lenet_float_realWeights[0],
    "MC4_8x8lenet_fixed_realWeight_ratio": MC4_8x8lenet_fixed_realWeights[1] / MC4_8x8lenet_fixed_realWeights[0],
    "darknet64_float_ratio": darknet64_float_results[1] / darknet64_float_results[0],
    "darknet64_fixed_ratio": darknet64_fixed_results[1] / darknet64_fixed_results[0],
    "vgg_float_ratio": vgg_float_results[1] / vgg_float_results[0],
    "vgg_fixed_ratio": vgg_fixed_results[1] / vgg_fixed_results[0]
}
