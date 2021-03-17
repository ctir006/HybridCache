import numpy as np
from kmeans_clustering import kmeans_clustering
from collections import defaultdict
import time


###### The following code is to load the data into the arrays  ######
def load_data():
    fileName1="predicted"
    fileName2="Actual"
    predicted_vals=[]
    actual_vals=[]
    for i in range(50):
        filename=fileName1+str(i+1)+".txt"
        Predicted=np.loadtxt(filename,dtype=int)
        predicted_vals.append(Predicted)
        filename=fileName2+str(i+1)+".txt"
        Actual=np.loadtxt(filename,dtype=int)
        actual_vals.append(Actual)
    return predicted_vals,actual_vals
    
def generate_transmission_delays(number_of_SB):
    n=number_of_SB
    no_delays=(n*(n-1))//2
    random_delays = np.random.uniform(10,30,no_delays)
    transmission_delays = [[0 for _ in range(number_of_SB)] for i in range(number_of_SB)]
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
    transmission_delays_sbs=[[0 for _ in range(number_of_SB)]for i in range(number_of_SB)]
    for i in range(number_of_SB):
        for j in range(number_of_SB):
            transmission_delays_sbs[i][j]=x[i][j]+y[i][j]
    return transmission_delays_sbs        
    
def get_popularity(values):
    sum=np.sum(values)
    return values/sum
    
def get_prediction_probabilities(data):
    predicted_probabilities=[]
    for sb in data:
        predicted_probabilities.append(sb/sb.sum(axis=1)[:,None])
    return predicted_probabilities
    
if __name__=="__main__":
    start_time=time.time()
    np.random.seed(0)
    each_content_size=20
    time_steps=20   #Number of hours for the simulation
    number_of_SB=10     #Number of Base Stations in the simulation
    cache_sizeSB=20     #Cache Size small base stations
    scache_sizeCC=80    #Cache size cloud cache 
    no_clusters=4   #k value for k-means clustering
    edge_caches=defaultdict(lambda:[])
    predicted_vals,actual_vals=load_data() # Shape of predicted_vals,actual_vals is # 50 X 20 X 1611
    predicted_probabilities=get_prediction_probabilities(predicted_vals)
    transmission_delays_sbs=generate_transmission_delays(number_of_SB)
    transmission_delays_remoteServers=np.random.uniform(60,100,number_of_SB)
    BX=[[0 for _ in range(1611)] for i in range(number_of_SB)]
    BS_labels=None

    all_delay_1=[]
    all_delay_2=[]
    all_hit_ratios=[]
    all_backhaul_congestions=[]
    for time_step in range(time_steps):
        if time_step%24 == 0:             # Do clustering of Base stations once per day
            BX=[[0 for _ in range(1611)] for i in range(number_of_SB)]
            edge_caches.clear()
            data=[]
            for BS in range(number_of_SB):
                data.append(predicted_vals[BS][time_step])
            obj=kmeans_clustering(data,no_clusters)
            Clusters,BS_labels=obj.kmeans() # The k-means function returns cluster number and base stations associated with that base station
            ######## Proactive Caching Process begins here ########
            print(Clusters)
            for cluster in range(no_clusters):      # Iterating through each cluster
                base_stations = Clusters[cluster]
                if len(base_stations)==1:           # If the cluster has only one base station
                    base_station = base_stations[0]
                    popularity = get_popularity(predicted_vals[base_station][time_step])
                    cached_contentIDs = popularity.argsort()[-cache_sizeSB:][::-1]
                    edge_caches[base_station].extend(cached_contentIDs)
                    for cached_contentID in cached_contentIDs:
                        BX[base_station][cached_contentID]=1
                else:                               # If the cluster has more than one base station
                    bs_object_popularities = defaultdict(lambda:[])
                    print("Caching at cluster : ",cluster)
                    print("Filling caches at the base stations : ",base_stations)
                    for base_station in base_stations:
                        bs_object_popularities[base_station].extend(get_popularity(predicted_vals[base_station][time_step]))
                    counter=0
                    contents=[(_,b) for _ in range(1611) for b in base_stations]
                    cache_count=0
                    while cache_count<len(base_stations)*cache_sizeSB:
                    #for _ in range(len(base_stations)*cache_sizeSB):
                        max=0
                        content_id,SB=None,None
                        for n,k in contents:
                            #for k in range(len(base_stations)):
                                bs=k
                                prev=0
                                for b_station in base_stations:
                                    prev+=transmission_delays_remoteServers[b_station]*np.sum(np.multiply(BX[b_station],bs_object_popularities[b_station]))
                                cur=0
                                BX[bs][n]=1
                                for b_station in base_stations:
                                    cur+=transmission_delays_remoteServers[b_station]*np.sum(np.multiply(BX[b_station],bs_object_popularities[b_station]))
                                BX[bs][n]=0
                                if (cur-prev)>max:
                                    max=(cur-prev)
                                    content_id=n
                                    SB=bs
                        if len(edge_caches[SB])<cache_sizeSB:
                            if (content_id,SB) in contents:
                                contents.remove((content_id,SB))
                            BX[SB][content_id]=1
                            edge_caches[SB].append(content_id)
                            cache_count+=1
                            print("Counter ",counter,"Item ",content_id," Cached at Base station ",SB)
                            counter+=1
                        else:
                            contents.remove((content_id,SB))
                    for base_station in base_stations:
                        print(base_station,edge_caches[base_station])
            print("--- %s seconds ---" % (time.time() - start_time))
        else:
            hits=0
            miss=0
            backhaul_load=0
            delay_1=0
            delay_2=0
            for BS in range(number_of_SB):
                requests=actual_vals[BS][time_step-1]
                for c_id,request in enumerate(requests):
                    f=0
                    for in_cluster_bs in range(number_of_SB):
                        if c_id in edge_caches[in_cluster_bs]:
                            hits+=request
                            f=1
                            if in_cluster_bs==BS:
                                backhaul_load+=0
                                delay_1+=0
                                delay_2+=0
                            else:
                                backhaul_load+=0
                                delay_1+=transmission_delays_sbs[BS][in_cluster_bs]
                                delay_2+=transmission_delays_sbs[BS][in_cluster_bs]
                            break
                    if f==1:
                        continue
                    else:                           # If the requested content is not present then reactive cache replacement come into picture
                        miss+=request
                        backhaul_load+=(request*each_content_size)
                        delay_1+=(request*transmission_delays_remoteServers[BS])
                        delay_2+=transmission_delays_remoteServers[BS]
                        replace=[]
                        for sc in edge_caches[BS]:
                            cache_sc=0
                            for bs in range(number_of_SB):
                                req=None
                                containing_bs=[]
                                containing_bs_tdelay=[]
                                for all_bs in range(number_of_SB):
                                    if c_id in edge_caches[all_bs]:
                                        containing_bs.append(all_bs)
                                for each_bs in containing_bs:
                                    containing_bs_tdelay.append(transmission_delays_sbs[bs][each_bs])
                                if containing_bs_tdelay:
                                    req=containing_bs[containing_bs_tdelay.index(min(containing_bs_tdelay))]
                                    temp=transmission_delays_sbs[bs][req]-transmission_delays_sbs[bs][BS]
                                    if temp>0:
                                        cache_sc+=temp
                            evict_sc=0
                            for bs in range(number_of_SB):
                                containing_bs=[]
                                containing_bs_tdelay=[]
                                for all_bs in range(number_of_SB):
                                    if all_bs==BS:
                                        continue
                                    if sc in edge_caches[all_bs]:
                                        containing_bs.append(all_bs)
                                for each_bs in containing_bs:
                                    containing_bs_tdelay.append(transmission_delays_sbs[bs][each_bs])
                                if containing_bs_tdelay:
                                    b=containing_bs[containing_bs_tdelay.index(min(containing_bs_tdelay))]
                                    temp=transmission_delays_sbs[bs][b]-transmission_delays_sbs[bs][BS]
                                    if temp>0:
                                        evict_sc+=temp
                            replace_temp=(cache_sc*(predicted_probabilities[BS][time_step][c_id]))-(evict_sc*(predicted_probabilities[BS][time_step][sc]))
                            replace.append(replace_temp)
                        replace=np.array(replace)
                        if np.max(replace)<0:              # If all the replace values are negative so no need of replacing the current requested content
                            continue                
                        else:                           # If not we should replace with the content which gives the max network delay reduction
                            edge_caches[BS][np.argmax(replace)]=c_id                                                           
            print("Time step ",time_step-1,"Hit ratio : ",hits/(hits+miss))
            all_hit_ratios.append(hits/(hits+miss))
            print("Time step ",time_step-1,"delay 1 (ms) : ",delay_1)
            all_delay_1.append(delay_1)
            print("Time step ",time_step-1,"delay 2 (ms) : ",delay_2)
            all_delay_2.append(delay_2)
            print("Time step ",time_step-1,"backhaul load (MB) : ",backhaul_load)
            all_backhaul_congestions.append(backhaul_load)
    print("Hit ratios : ",all_hit_ratios)
    print("Delay 1 : ",all_delay_1)
    print("Delay 2 : ",all_delay_2)
    print("Backhaul congestion : ",all_backhaul_congestions)
    print("Average hit ratio : ",sum(all_hit_ratios)/len(all_hit_ratios)) 
    print("Average delay_1 : ",sum(all_delay_1)/len(all_delay_1))
    print("Average delay_2 : ",sum(all_delay_2)/len(all_delay_2))
    print("Average backhaul_load : ", sum(all_backhaul_congestions)/len(all_backhaul_congestions))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
