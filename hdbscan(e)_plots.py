# -*- coding: utf-8 -*-
"""
======================================
Comparison of HDBSCAN(epsilon), HDBSCAN(eom), 
DBSCAN and OPTICS on different datasets 
======================================
"""
print(__doc__)

import os
import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import folium

fileLocation = os.path.realpath(__file__) 
workspace = os.path.dirname(fileLocation) #directory of this .py script

def load_data(name):
   
    data = np.loadtxt(os.path.join(workspace, name))
    return(data[:,:2], data[:, 2])
    
def create_aniso_data():
    # source: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    n_samples = 150
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return (X_aniso, y)

def plot_clustering(X, colors, labels, axis):
    unique_labels = set(labels)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0.6, 0.6, 0.6, 0]
            axis.plot(X[labels == k, 0], X[labels == k, 1], 'o', markerfacecolor=col,
                          markeredgecolor='k', markersize=6)
            
    #workaround: make sure that noise is plotted first and thus stays in the background  
    for k, col in zip(unique_labels, colors):
        if k != -1:
            axis.plot(X[labels == k, 0], X[labels == k, 1], 'o', markerfacecolor=col,
                          markeredgecolor='k', markersize=6)
    
    axis.locator_params(nbins=4)
    
def count_clusters(labels):
    return len(set(labels)) - (1 if -1 in labels else 0)    
    
def colorize(labels, dataset_name, algorithm):

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    #customizations to make sure that clusters are well distinguishable 
    if dataset_name == "DS1":
        
         if algorithm == "DBSCAN":
             colors[0] = (0.2, 0.6, 0.9, 1) 
             colors[4]=  (0.9, 0.9, 0.4, 1)
             
         elif algorithm == "OPTICS":
             colors[27] = (0.9, 0.9, 0.4, 1)
             colors[29] = (0.8, 0.3, 0.5, 1) 
             colors[30] = (0.4, 0.4, 0.8, 1) 
             colors[32] = (0.3, 0.7, 0.4, 1) 
             colors[33] = (0.2, 0.6, 0.9, 1) 
             
         elif algorithm == "HDBSCAN(epsilon)":
             colors[4] = (0.3, 0.8, 0.4, 1)
             colors[7] = (1, 0.5, 0.6, 1)
             
    elif dataset_name == "DS2":
        
         if algorithm == "HDBSCAN(epsilon)": 
            colors[4]=  (0.2, 0.6, 0.9, 1) 
            
         elif algorithm == "OPTICS": 
            colors[16] = (0.4, 0.3, 0.7, 1)
            colors[20] = (0.3, 0.7, 0.4, 1)
            colors[30] = (0.2, 0.6, 0.9, 1) 
           
    if  algorithm == "HDBSCAN(eom)" and (dataset_name == "DS1" or dataset_name == "DS2"): 
        
            colors[0] = (0.2, 0.6, 0.9, 1) 
            colors[1] = (1, 0.5, 0.6, 1) 
            colors[2] = (0.0, 0.3, 0.6, 1)
            colors[3] = (0.9, 0.9, 0.4, 1)
            colors[4] = (0.3, 0.7, 0.4, 1) 
            colors[5] = (0.9, 0.2, 0.4, 1)  
            colors[6] = (0.1, 0.4, 0.4, 1)
            colors[7] = (0.2, 0.5, 0.9, 1) 
            
    return colors
     
    
def cluster_and_plot(dataset):

    data = dataset[0]
    name = dataset[1]
    
    minPts = 4
    X, true_labels = data
    # normalize dataset for easier parameter selection
    # note: for un-normalized data sets, run HDBSCAN(e) with 21000 for DS1 and 0.00003 for DS2 
    X = StandardScaler().fit_transform(X)

    hdbe_epsilons = {"jain": 0.315, "spiral": 0.3, "flame": 0, "aniso": 0.3, "DS1": 0.2, "DS2": 0.1} 
    db_epsilons = {"jain": 0.315, "spiral": 0.3, "flame": 0.28, "aniso": 0.3, "DS1": 0.45, "DS2": 0.38}

    # HDBSCAN(eom)
    hdb = hdbscan.HDBSCAN(min_cluster_size=minPts, cluster_selection_method="eom").fit(X)
    hdb_labels = hdb.labels_
    
    # HDBSCAN(epsilon) 
    hdbe = hdbscan.HDBSCAN(min_cluster_size=minPts, cluster_selection_epsilon=hdbe_epsilons[name], cluster_selection_method="eom").fit(X) 
    hdbe_labels = hdbe.labels_
    
    # DBSCAN
    db = DBSCAN(min_samples=minPts, eps=db_epsilons[name]).fit(X)
    db_labels = db.labels_
    
    # OPTICS
    optics_clusterer = OPTICS(min_samples=minPts, xi=0.05).fit(X) 
    opt_labels = optics_clusterer.labels_
    
    # colorize
    true_colors = colorize(true_labels, name, "true") 
    hdb_colors = colorize(hdb_labels, name, "HDBSCAN(eom)")
    opt_colors = colorize(opt_labels, name, "OPTICS")
    db_colors = colorize(db_labels, name, "DBSCAN")
    hdbe_colors = colorize(hdbe_labels, name, "HDBSCAN(epsilon)") 
        
    # plot
    fig = plt.figure(figsize=(12,13)) 
    true_axis = fig.add_subplot(3,2,1)
    hdb_axis = fig.add_subplot(3,2,3)
    opt_axis = fig.add_subplot(3,2,4)
    db_axis = fig.add_subplot(3,2,5) 
    hdbe_axis = fig.add_subplot(3,2,6)
    fig.subplots_adjust(hspace=.5)
                        
    plot_clustering(X, true_colors, true_labels, true_axis)
    plot_clustering(X, hdb_colors, hdb_labels, hdb_axis)
    plot_clustering(X, hdbe_colors, hdbe_labels, hdbe_axis)
    plot_clustering(X, db_colors, db_labels, db_axis)
    plot_clustering(X, opt_colors, opt_labels, opt_axis)
    
    true_axis.set_title('Number of clusters: %d' % count_clusters(true_labels))
    hdb_axis.set_title('HDSBCAN(eom)\nEstimated number of clusters: %d' % count_clusters(hdb_labels))
    hdbe_axis.set_title('HDBSCAN(epsilon)\nEstimated number of clusters: %d' % count_clusters(hdbe_labels))
    db_axis.set_title('DBSCAN\nEstimated number of clusters: %d' % count_clusters(db_labels))
    opt_axis.set_title('OPTICS\nEstimated number of clusters: %d' % count_clusters(opt_labels))
    
    plt.show()
    
    # statistics
    db_noise = list(db_labels).count(-1)
    hdb_noise = list(hdb_labels).count(-1)
    hdbe_noise = list(hdbe_labels).count(-1)
    opt_noise = list(opt_labels).count(-1)
    
    print("=====================================")
    print("Results for data set '{}':".format(name))
    print("=====================================")
    print("DBSCAN with epsilon = {}".format(db_epsilons[name]))
    print("Adjusted Rand Index: {0:.2f}".
          format(metrics.adjusted_rand_score(true_labels, db_labels)))
    print("Covered: {0:.2f}".format(1-db_noise/len(db_labels)))
    print("---------------------")
    print("OPTICS")
    print("Adjusted Rand Index: {0:.2f}".
          format(metrics.adjusted_rand_score(true_labels, opt_labels)))
    print("Covered: {0:.2f}".format(1-opt_noise/len(opt_labels)))
    print("---------------------")
    print("HDBSCAN(eom)")
    print("Adjusted Rand Index: {0:.2f}".
          format(metrics.adjusted_rand_score(true_labels, hdb_labels)))
    print("Covered: {0:.2f}".format(1-hdb_noise/len(hdb_labels)))
    print("---------------------")
    print("HDBSCAN(epsilon) with epsilon = {}".format(hdbe_epsilons[name]))
    print("Adjusted Rand Index: {0:.2f}".
          format(metrics.adjusted_rand_score(true_labels, hdbe_labels)))
    print("Covered: {0:.2f}".format(1-hdbe_noise/len(hdbe_labels)))
    print("=====================================")


def cluster_coords(coords):
    
    print("Clustering {} GPS data points ...".format(len(coords)))
    X = np.radians(coords)
    kms_per_radian = 6371.0088 
    meters = 3  # use this to tune the threshold
    epsilon = meters/1000 / kms_per_radian 
    
    minPts = 4
    print("... with HDSCAN(eom)...")
    hdb = hdbscan.HDBSCAN(min_cluster_size=minPts, metric = "haversine", cluster_selection_method="eom").fit(X)
    hdb_labels = hdb.labels_
    
    print("... with HDBSCAN(epsilon)...")
    hdbe = hdbscan.HDBSCAN(min_cluster_size=minPts, metric = "haversine", cluster_selection_epsilon=epsilon, cluster_selection_method="eom").fit(X) 
    hdbe_labels = hdbe.labels_
    
    print("... with DBSCAN...")
    db = DBSCAN(min_samples=minPts, metric = "haversine", eps=epsilon).fit(X)
    db_labels = db.labels_
    
    print("... with OPTICS...")
    optics_clusterer = OPTICS(min_samples=minPts, xi=0.05).fit(X) 
    opt_labels = optics_clusterer.labels_
    
    plotMap(coords, hdbe_labels) #choose which results to plot in a web map
    
    print("Clusters found by HDBSCAN(eom): {}".format(count_clusters(hdb_labels)))
    print("Clusters found by DBSCAN with an epsilon of {} meters: {}".format(meters, count_clusters(db_labels)))
    print("Clusters found by OPTICS: {}".format(count_clusters(opt_labels)))
    print("Clusters found by HDBSCAN(epsilon) with a threshold of {} meters: {}".format(meters, count_clusters(hdbe_labels)))
    print("Check output at {}".format(workspace+"/gps_clustering_results.html"))
    
    
def plotMap(locations, labels):
    
    clusters = set(labels)
    colors = [plt.cm.prism(each)
              for each in np.linspace(0, 1, len(clusters))]

    #hard coded customizations to make clusters better distinguishable 
   
    colors[0] = (0.4, 0.3, 0.7)
    colors[2] = (0.4, 0.4, 1) 
    colors[3] = (0.8, 0.3, 0.5)
    colors[4] = (1, 0.3, 0.)
    colors[5] = (0.4, 0.5, 0.8)
    colors[7] = (1, 0.5, 0.6)
    colors[9] = (0.1, 0.7, 0.8)
    colors[10] = (0.6, 0.3, 0.5)
    colors[11] = (0.4, 0.3, 0.1)
    if len(colors) > 55:
        colors[16] = (1, 0.3, 0.5)
        colors[17] = (0.4, 0.4, 0.7)
        colors[19] = (0, 0.5, 0.4)
        colors[20] = (0, 0.3, 0.6)
        colors[21] = (0, 0.3, 0.6)
        colors[22] = (0.4, 0.8, 0.4)
        colors[23] = (0.6, 0.2, 0.4)
        colors[24] = (0.8, 0.4, 0.7)
        colors[25] = (0.9, 0.2, 0.4)
    if len(colors) > 246:
         colors[229] =  (0.4, 0.8, 0.4)
         colors[230] =  (0.6, 0.2, 0.4)
         colors[231] =  (0.8, 0.4, 0.7)
         colors[238] =  (0.4, 0.3, 0.1)
         colors[243] =  (0, 0.3, 0.6)
         colors[245] = (0.9, 0.2, 0.4)
         colors[246] = (0.4, 0.4, 0.7)
         
    
    map = folium.Map(
        location=[51.911016, 10.421581],
        tiles='Stamen Toner',
        zoom_start=18,
        max_zoom=24
    )
    
    cluster_points = []
    noise_points = []
    for index, point in enumerate(locations):
        cluster_id = labels[index]
        if cluster_id == -1:
            marker = folium.CircleMarker(location=[point[0], point[1]], radius=5, popup = "Noise", weight=1.5, color="#000000", fill_color= "#BDBDBD", opacity = 0.5, fill_opacity = 1.0, fill=True)
            noise_points.append(marker)
        else: 
            hexcolor = mcolors.rgb2hex(colors[cluster_id])
            marker = folium.CircleMarker(location=[point[0], point[1]], popup = str(cluster_id), radius=6, weight=2.5, color="#000000", fill_color= hexcolor, opacity = 1.0, fill_opacity = 1.0, fill=True)
            cluster_points.append(marker)                             

    for noise in noise_points:
        noise.add_to(map)
    for point in cluster_points:
        point.add_to(map)
    
    map.save(os.path.join(workspace, "gps_clustering_results.html"))
    
if __name__ == '__main__':
    
    flame = (load_data("data/flame.txt"), "flame")
    spiral = (load_data("data/spiral.txt"), "spiral")
    jain = (load_data("data/jain.txt"), "jain")
    aniso = (create_aniso_data(), "aniso")
    ds1 = (load_data("data/synthetic_ds1.txt") , "DS1")
    ds2 = (load_data("data/synthetic_ds2.txt") , "DS2")
    
    #START HERE: choose a data set to plot
    cluster_and_plot(ds1)
    
    #Additionally, create leaflet web map with GPS sample data set
    gps_coords = np.loadtxt(os.path.join(workspace, "data/gps_sample_data.txt"))
    cluster_coords(gps_coords)
    print("=====================================")