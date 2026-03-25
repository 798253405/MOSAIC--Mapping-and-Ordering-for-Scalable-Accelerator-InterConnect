# Adjusting the plot to separate float and fixed for each category
import numpy as np
import matplotlib.pyplot as plt
#%%
lenet_float_results = np.array([ 21178213, 18525122 ,16199367 ])
lenet_fixed_results  = np.array([ 5849232, 5373406, 4857909])
#lenet_float_results = np.array([ 21771412, 18218297 ,16789674])
#lenet_fixed_results  = np.array([ 5616286, 4889065, 4693078])
# Packet id: 8094 YZGlobalFlit_id 74554 YZGlobalFlitPass 202566 YZGlobalRespFlitPass 158610 yzWeightCollsionInRouterCountSum 5840312 yzWeightCollsionInNICountSum 74554 yzFlitCollsionCountSum 202566 tempRouterNetWholeFlipCount 21458990 tempRouterNetWholeFlipCount_fix35 5840312
#Packet id: 8094 YZGlobalFlit_id 74554 YZGlobalFlitPass 202566 YZGlobalRespFlitPass 158610 yzWeightCollsionInRouterCountSum 3741965 yzWeightCollsionInNICountSum 74554 yzFlitCollsionCountSum 202566 tempRouterNetWholeFlipCount 14590875 tempRouterNetWholeFlipCount_fix35 3741965
lenet_float_resultsRealWeight = np.array([ 21458990,  17473331 ,14590875])
lenet_fixed_resultsRealWeight  = np.array([ 5840312, 4803903,  3741965])

# memnode 8 , 8x8
MC8_8x8lenet_float_results = np.array([ 21135707,18579938 ,16210338 ])
MC8_8x8lenet_fixed_results  = np.array([ 5844551, 5384113, 4856594])
MC8_8x8lenet_float_realWeights = np.array([ 21362724,17431893  ,14558921 ])
MC8_8x8lenet_fixed_realWeights  = np.array([  5822190, 4791255,3730976])

# memnode 4 , 8x8
MC4_8x8lenet_float_results = np.array([ 26128676,22909347 ,20016888 ])
MC4_8x8lenet_fixed_results  = np.array([ 7230708,  6636816, 5999163])
MC4_8x8lenet_float_realWeights = np.array([  26586230 , 21646607 ,18244119 ])
MC4_8x8lenet_fixed_realWeights  = np.array([  7229246,  5971260, 4707240])
# Data for plotting
#categories = ['Baseline', 'Weight-based ordering', 'Seperately ordering','Baseline', 'Weight-based ordering', 'Seperately ordering','Baseline', 'Weight-based ordering', 'Seperately ordering','Baseline', 'Weight-based ordering', 'Seperately ordering','Baseline', 'Weight-based ordering', 'Seperately ordering']
#categories= ['Baseline', 'Weight-based ordering', 'Seperately ordering']
categories= ['O0', 'O1', 'O2']
bar_width = 0.3
index = np.arange(3)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot 1: Float results
ax.bar(index - bar_width/2, lenet_float_results, bar_width, label='MC2 4X4 float-32 random')
ax.bar(index + bar_width/2, lenet_fixed_results, bar_width, label='MC2 4X4 fixed-8 random')

# Plot 2: Real Weights results
ax.bar(index - bar_width/2 + len(categories), lenet_float_resultsRealWeight, bar_width, label='MC2 4X4 float-32 trained')
ax.bar(index + bar_width/2 + len(categories), lenet_fixed_resultsRealWeight, bar_width, label='MC2 4X4 fixed-8 trained')

# Plot 3: MC8 8x8 Float and Fixed results
ax.bar(index - bar_width/2 + 2*len(categories), MC8_8x8lenet_float_results, bar_width, label='MC8 8x8 float-32 random')
ax.bar(index + bar_width/2 + 2*len(categories), MC8_8x8lenet_fixed_results, bar_width, label='MC8 8x8 fixed-8 random')

# Plot 4: MC8 8x8 Real Weights results
ax.bar(index - bar_width/2 + 3*len(categories), MC8_8x8lenet_float_realWeights, bar_width, label='MC8 8x8 float-32 trained')
ax.bar(index + bar_width/2 + 3*len(categories), MC8_8x8lenet_fixed_realWeights, bar_width, label='MC8 8x8 fixed-8 trained')

# Plot 5: MC4 8x8 Float and Fixed results
ax.bar(index - bar_width/2 + 4*len(categories), MC4_8x8lenet_float_results, bar_width, label='MC4 8x8 float-32 random')
ax.bar(index + bar_width/2 + 4*len(categories), MC4_8x8lenet_fixed_results, bar_width, label='MC4 8x8 fixed-8 random')

# Plot 6: MC4 8x8 Real Weights results
ax.bar(index - bar_width/2 + 5*len(categories), MC4_8x8lenet_float_realWeights, bar_width, label='MC4 8x8 float-32 trained',color='orange')
ax.bar(index + bar_width/2 + 5*len(categories), MC4_8x8lenet_fixed_realWeights, bar_width, label='MC4 8x8 fixed-8 trained',color = 'lightgreen')

# Adding labels and title
#ax.set_xlabel('Categories')
ax.set_ylabel('Bit trainstions')
#ax.set_title('Results for Float and Fixed across different sets')
#xticks = ['O0','O0', 'O1', 'O2','O0', 'O1', 'O2','O0', 'O1', 'O2','O0', 'O1', 'O2','O0', 'O1', 'O2','O0', 'O1', 'O2']
#ax.set_xticklabels(xticks)
#ax.set_xticks(xticks)
ax.set_xticklabels([])
ax.set_xlabel('Different ordering configurations')
#ax.set_xticklabels(categories)
ax.legend()

# Show plot
plt.tight_layout()
plt.show()
