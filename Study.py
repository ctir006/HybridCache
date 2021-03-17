import numpy as np
number_of_SB=50

l=[0.7396896703343275,0.7179659018165588,0.7127892583980735,0.7188518781006379,0.7150689606854815,0.7204336805936065,0.7157778826490483,0.711372593819615,0.7064937570991892,0.6871992379920138,0.6839810704306452,0.6883224574084811,0.6839211801680359,0.6861023219984295,0.6882159813809154,0.6749692979018056,0.6933584686774942,0.6962764013587927,0.7109168261700067]
m=[0.6345555648066127,0.6261854585913391,0.6328929836995039,0.6273821554685953,0.6328332408327176,0.6296373883984239,0.625840256232002,0.6210820150952333,0.6019863090910407,0.6044339508664057,0.6140058523828128,0.6094876216587225,0.6146480791060828,0.6142411171450737,0.6028520289514253,0.6164608882996354,0.614750915690069,0.6305776400109152]

print(sum(l)/len(l))
print(sum(m)/len(m))
quit()

# l = [1155, 398, 748, 1167, 1259, 760, 402, 1242, 406, 401, 408, 0, 409, 1227, 1278, 410, 450, 400, 761, 405,1155, 1180, 751, 0, 
# 1259, 750, 1158, 31, 1167, 764, 19, 24, 1156, 403, 1165, 5, 412, 405, 1186, 398,749, 1178, 400, 511, 405, 406, 750, 1186, 0, 1180, 
# 1, 1156, 412, 762, 399, 1158, 24, 1173, 1242, 411,1167, 401, 1259, 783, 398, 0, 1232, 511, 1173, 1186, 750, 24, 1180, 2, 837, 62, 
# 1171, 415, 1178, 757,749, 748, 405, 1155, 398, 400, 399, 1173, 1186, 1177, 1167, 755, 762, 1158, 1278, 410, 1259, 0, 5, 1232,748, 
# 749, 1259, 511, 1186, 402, 1167, 760, 1194, 406, 0, 408, 1158, 806, 1173, 1180, 398, 751, 754, 1178,400, 1167, 1180, 749, 511, 398, 
# 1175, 837, 1155, 1153, 0, 1158, 1178, 1154, 408, 1160, 16, 401, 806, 115,0, 398, 1167, 400, 765, 1195, 408, 511, 748, 1155, 760, 
# 1153, 1160, 1186, 39, 1158, 1173, 762, 409, 401,0, 1167, 748, 751, 750, 1155, 5, 1186, 399, 398, 749, 1173, 402, 400, 1177, 19, 757, 1160, 405, 1154,
# 412, 1155, 398, 749, 748, 1248, 1167, 1154, 1186, 13, 511, 405, 0, 1173, 1195, 31, 1153, 7, 750, 4]

print(len(l))
print(len(set(l)))
quit()


n=number_of_SB
no_delays=(n*(n-1))//2
random_delays = np.random.uniform(10,30,no_delays)
transmission_delays = [[0 for _ in range(number_of_SB)] for i in range(number_of_SB)]
for i in transmission_delays:
    print(i[:10])
pos=0
for i in range(number_of_SB):
    for j in range(number_of_SB):
        if i>=j:
            continue
        else:
            transmission_delays[i][j]=random_delays[pos]
            pos+=1
x = [[row[i] for row in transmission_delays] for i in range(len(transmission_delays[0]))]
y = transmission_delays
delays=[[0 for _ in range(number_of_SB)]for i in range(number_of_SB)]
for i in range(number_of_SB):
    for j in range(number_of_SB):
        delays[i][j]=x[i][j]+y[i][j]
for i in delays:
    print(i[:10])
print(no_delays)












#######################################################################################
# import numpy as np
# from sklearn.preprocessing import scale
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits


# digits = load_digits()
# data = digits.data

# print(len(data),len(data[0]))

# x=[]
# y=[]
# for noclusters in range(1,20):
    # x.append(noclusters)
    # kmeans = KMeans(init='k-means++', n_clusters=noclusters).fit(scale(data))
    # y.append(kmeans.inertia_)
    # print(kmeans.inertia_)
    
# plt.figure()
# plt.plot(x,y)
# plt.xlabel("Number of cluster")
# plt.ylabel("SSE")
# plt.show()
# print(len(data),len(data[0]))