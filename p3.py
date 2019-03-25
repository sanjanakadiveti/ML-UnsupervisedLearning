from sklearn.cluster import KMeans as KM
from sklearn.mixture import GaussianMixture as EM
from sklearn.decomposition.pca import PCA as PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as RandomProjection
from sklearn.feature_selection import SelectKBest as best
from sklearn.feature_selection import chi2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from scipy.stats import kurtosis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import os
from mpl_toolkits.mplot3d import Axes3D
import time

def km(tx, ty, rx, ry, reduced_data, add="", times=5, dataset="", alg=""):
    processed = []
    adj_rand = []
    v_meas = []
    mutual_info = []
    adj_mutual_info = []
    sil = []
    inertia = []
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}
        clf = KM(n_clusters=i)
        clf.fit(tx)
        test = clf.predict(tx)
        result = clf.predict(rx)

        adj_rand.append(metrics.adjusted_rand_score(ry.ravel(), result))
        v_meas.append(metrics.v_measure_score(ry.ravel(), result))
        mutual_info.append(metrics.fowlkes_mallows_score(ry.ravel(), result))
        adj_mutual_info.append(metrics.homogeneity_score(ry.ravel(), result))
        inertia.append(clf.inertia_)
    plots = [adj_rand, v_meas, mutual_info, adj_mutual_info]
    plt.title(dataset+": KM Clustering measures - "+alg)
    plt.xlabel('Number of clusters')
    plt.ylabel('Score value')
    plt.plot(range(2,times), adj_rand, label="Adjusted Random")
    plt.plot(range(2,times), v_meas, label="V Measure")
    plt.plot(range(2,times), mutual_info, label = "Fowlkes Mallows Score")
    plt.plot(range(2,times), adj_mutual_info, label="Homogeneity Score")
    plt.legend()
    plt.ylim(ymin=-0.05, ymax=1.05)
    plt.savefig("KMeansMetric"+dataset+"_"+alg+".png")

    plt.figure()
    plt.title(dataset+": KMeans Inertia - "+alg)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.plot(range(2,times), inertia)
    plt.savefig("KM-Inertia-"+dataset+"-"+alg+".png")

    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(tx, td, 1)
    newrx = np.append(rx, rd, 1)

    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
        # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    best_clusterer = KM(n_clusters=4)
    best_clusterer.fit(X)
    Z = best_clusterer.predict(X)
    print(len(Z))
    print(len(X))
    plt.figure(1)
    plt.clf()
    colors = ['r', 'g', 'b', 'y', 'c', 'm','#eeefff', '#317c15', '#4479b4', '#6b2b9c',
'#63133b', '#6c0d22', '#0c7c8c', '#67c50e','#c5670e', '#946c47', '#58902a', '#54b4e4',
'#e4549e', '#2b2e85'  ]
    for i in range(0, len(X)):
        plt.plot(X[i][0], X[i][1], marker='.', color=colors[Z[i]], markersize=2)
    #plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = best_clusterer.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='k', zorder=10)
    plt.title('K-means Clusters ' + alg)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    kmeans = KM(n_clusters=3)
    kmeans.fit(tx)
    result=pd.DataFrame(kmeans.transform(tx), columns=['KM%i' % i for i in range(3)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['KM0'], result['KM1'], result['KM2'], c=my_color, cmap="Dark2_r", s=60)
    plt.show()
    reduced_data = PCA(n_components=2).fit_transform(tx)
    kmeans = KM(n_clusters=4)
    kmeans.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(dataset + ': K-means clustering (' + alg + '-reduced data)\n'
              'Centroids are marked with a white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    checker = KM(n_clusters=2)
    ry = ry.reshape(-1,1)
    checker.fit(ry)
    truth = checker.predict(ry)
    clusters = {x:[] for x in range(4)}
    clf = KM(n_clusters=4)
    clf.fit(tx)  #fit it to our data
    test = clf.predict(tx)
    result = clf.predict(rx)  # and test it on the testing set
    for index, val in enumerate(result):
        clusters[val].append(index)
    mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(4)}
    processed = [mapper[val] for val in result]
    print(sum((processed-truth)**2) / float(len(ry)))
    clf = KM(n_clusters=times)
    clf.fit(tx)  #fit it to our data
    test = clf.predict(tx)
    result = clf.predict(rx)
    checker = KM(n_clusters=times)
    ry = ry.reshape(-1,1)
    checker.fit(ry)
    truth = checker.predict(ry)
    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    newtx = np.append(td)
    newrx = np.append(rd)
    myNN(test, ty, result, ry, alg="KM_"+alg)
    nn(newtx, ty, newrx, ry, add="onKM"+add)

def em(tx, ty, rx, ry, reduced_data, add="", times=5, dataset="", alg=""):
    clf = EM(n_components=times)
    clf.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # centroids = clf.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    plt.title(dataset + ': EM clustering (' + alg + '-reduced data)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    clf = EM(n_components=times)
    clf.fit(tx)  #fit it to our data
    test = clf.predict(tx)
    result = clf.predict(rx)
    checker = EM(n_components=times)
    ry = ry.reshape(-1,1)
    checker.fit(ry)
    truth = checker.predict(ry)
    td = np.reshape(test, (test.size, 1))
    rd = np.reshape(result, (result.size, 1))
    # newtx = np.append(td)
    # newrx = np.append(rd)
    myNN(test, ty, result, ry, alg="EM_"+alg)
    errs = []
    scores = []
    # this is what we will compare to
    checker = EM(n_components=2)
    ry = ry.reshape(-1,1)
    checker.fit(ry)
    truth = checker.predict(ry)
    adj_rand = []
    v_meas = []
    mutual_info = []
    adj_mutual_info = []
    # so we do this a bunch of times
    for i in range(2,times):
        clusters = {x:[] for x in range(i)}
        # create a clusterer
        clf = EM(n_components=i)
        clf.fit(tx)  #fit it to our data
        test = clf.predict(tx)
        result = clf.predict(rx)  # and test it on the testing set
        for index, val in enumerate(result):
            clusters[val].append(index)
        mapper = {x: round(sum(truth[v] for v in clusters[x])/float(len(clusters[x]))) if clusters[x] else 0 for x in range(i)}
        processed = [mapper[val] for val in result]
        errs.append(sum((processed-truth)**2) / float(len(ry)))
        scores.append(clf.score(tx, ty))
        adj_rand.append(metrics.adjusted_rand_score(ry.ravel(), result))
        v_meas.append(metrics.v_measure_score(ry.ravel(), result))
        mutual_info.append(metrics.fowlkes_mallows_score(ry.ravel(), result))
        adj_mutual_info.append(metrics.homogeneity_score(ry.ravel(), result))
    # plot([0, times, min(scores)-.1, max(scores)+.1],[range(2, times), scores, "-"], "Number of Clusters", "Log Likelihood", dataset+": EM Log Likelihood - " + alg, dataset+"_EM_"+alg)

    # other metrics
    # names = ["Adjusted Random", "V Measure", "Mutual Info", "Adjusted Mutual Info"]
    plt.figure()
    plt.title(dataset+": EM Clustering measures - "+alg)
    plt.xlabel('Number of clusters')
    plt.ylabel('Score value')
    plt.plot(range(2,times),adj_rand, label="Adjusted Random")
    plt.plot(range(2,times),v_meas, label="V Measure")
    plt.plot(range(2,times),mutual_info, label = "Fowlkes Mallows Score")
    plt.plot(range(2,times),adj_mutual_info, label="Homogeneity Score")
    plt.legend()
    plt.savefig("EMMetrics"+dataset+"_"+alg+".png")

    kmeans = KM(n_clusters=2)
    kmeans.fit(reduced_data)

    Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(dataset + ': EM clustering (' + alg + '-reduced data)\n'
              'Centroids are marked with a white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def pca(tx, ty, rx, ry, dataset):
    ncomponents = tx[1].size/2
    compressor = PCA(n_components = ncomponents)
    xarr = []
    for i in range(0, ncomponents):
        xarr.append(i+1)
    compressor.fit(tx, y=ty)
    arr = compressor.explained_variance_
    plt.figure()
    plt.title('Phishing PCA Explained Variance')
    plt.rc('legend',**{'fontsize':10})
    plt.plot(xarr, arr, '-', label='explained variance')
    plt.legend()
    plt.ylabel('explained variance')
    plt.xlabel('number of components')
    plt.savefig("phishingPCAVar" + dataset + ".png")

    compressor = PCA(n_components = tx[1].size/2)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, add="wPCAtr", times=21, dataset=dataset, alg="PCA")
    em(newtx, ty, newrx, ry, PCA(n_components=2).fit_transform(tx), add="wPCAtr", times=9, dataset=dataset, alg="PCA")
    nn(newtx, ty, newrx, ry, add="wPCAtr")
    km(newtx, ty, newrx, ry, add="wPCAtr", times=10)
    myNN(newtx, ty, newrx, ry, "PCA")
    km(newtx, ty, newrx, ry, [], add="", times=4, dataset=dataset, alg="PCA")
    reduced_data = PCA(n_components=2).fit_transform(tx)
    em(tx, ty, rx, ry, reduced_data, add="", times=4, dataset=dataset, alg="PCA")
    pca = PCA(n_components=2)
    pca.fit(tx)
    result=pd.DataFrame(pca.transform(tx), columns=['PCA%i' % i for i in range(2)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(result['PCA0'], result['PCA1'], c=my_color, cmap="Dark2_r", s=60)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA on the phishing data set")
    plt.show()

def randproj(tx, ty, rx, ry, dataset):
    compressor = RandomProjection(tx[1].size/2)
    compressor = RandomProjection(tx[1].size/2)
    compressor.fit(tx, y=ty)
    pca = RandomProjection(2)
    pca.fit(tx)
    result=pd.DataFrame(pca.transform(tx), columns=['RP%i' % i for i in range(2)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='2d')
    ax = fig.add_subplot(111)
    ax.scatter(result['RP0'], result['RP1'], c=my_color, cmap="Dark2_r", s=60)
    ax.set_xlabel("RP1")
    ax.set_ylabel("RP2")
    ax.set_title("RP on the "+  dataset + " data set")
    plt.show()
    Store results of PCA in a data frame
    result=pd.DataFrame(compressor.transform(tx), columns=['ICA%i' % i for i in range(3)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['ICA0'], result['ICA1'], result['ICA2'], c=my_color, cmap="Dark2_r", s=60)

    xAxisLine = ((min(result['ICA0']), max(result['ICA0'])), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['ICA1']), max(result['ICA1'])), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(result['ICA2']), max(result['ICA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    ax.set_xlabel("RP1")
    ax.set_ylabel("RP2")
    ax.set_zlabel("RP3")
    ax.set_title("RP on the Car data set")
    plt.show()
    reduced_data = RandomProjection(2).fit_transform(tx)
    em(tx, ty, rx, ry, reduced_data, add="", times=4, dataset=dataset, alg="RP")
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, [], add="", times=4, dataset=dataset, alg="RP")
    em(newtx, ty, newrx, ry, RandomProjection(n_components=2).fit_transform(tx), add="wRPtr", times=9, dataset=dataset, alg="RandProj")
    # nn(newtx, ty, newrx, ry, add="wRPtr")
    myNN(newtx, ty, newrx, ry, "RandProj")


def kbest(tx, ty, rx, ry, dataset):
    reduced_data = best(k=2).fit_transform(tx, y=ty)
    em(tx, ty, rx, ry, reduced_data, add="", times=4, dataset=dataset, alg="KB")
    pca = best(k=2)
    pca.fit(tx, y=ty)

    result=pd.DataFrame(pca.transform(tx), columns=['KB%i' % i for i in range(2)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='2d')
    ax = fig.add_subplot(111)
    ax.scatter(result['KB0'], result['KB1'], c=my_color, cmap="Dark2_r", s=60)
    ax.set_xlabel("KB1")
    ax.set_ylabel("KB2")
    # ax.set_zlabel("PC3")
    ax.set_title("KB on the "+  dataset + " data set")
    plt.show()
    result=pd.DataFrame(compressor.transform(tx), columns=['ICA%i' % i for i in range(3)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['ICA0'], result['ICA1'], result['ICA2'], c=my_color, cmap="Dark2_r", s=60)

    ax.set_title("RP on the Phishing data set")
    plt.show()
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    em(newtx, ty, newrx, ry, [], add="", times=2, dataset=dataset, alg="KB")
    em(newtx, ty, newrx, ry, best(k=2).fit_transform(tx, y=ty), add="wKBtr", times=9, dataset=dataset, alg="K-Best")
    nn(newtx, ty, newrx, ry, add="wKBtr")
    myNN(newtx, ty, newrx, ry, "Kbest")

def ica(tx, ty, rx, ry, dataset):
    reduced_data = ICA(n_components=2).fit_transform(tx)
    em(tx, ty, rx, ry, reduced_data, add="", times=4, dataset=dataset, alg="ICA")
    x,y = tx.shape
    for i in range(0, y):
        print(kurtosis(tx[:,i], fisher=False))
    compressor = ICA(n_components = tx[1].size/2, max_iter=10000, tol=0.001)  # for some people, whiten needs to be off
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    # km(newtx, ty, newrx, ry, [], add="", times=7, dataset=dataset, alg="ICA")
    # Store results of PCA in a data frame

    pca = ICA(n_components=2)
    pca.fit(tx)
    result=pd.DataFrame(pca.transform(tx), columns=['PCA%i' % i for i in range(2)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='2d')
    ax = fig.add_subplot(111)
    ax.scatter(result['PCA0'], result['PCA1'], c=my_color, cmap="Dark2_r", s=60)
    plt.show()
    result=pd.DataFrame(compressor.transform(tx), columns=['ICA%i' % i for i in range(3)])
    my_color = pd.Series(ty).astype('category').cat.codes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['ICA0'], result['ICA1'], result['ICA2'], c=my_color, cmap="Dark2_r", s=60)

    xAxisLine = ((min(result['ICA0']), max(result['ICA0'])), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['ICA1']), max(result['ICA1'])), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(result['ICA2']), max(result['ICA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    ax.set_xlabel("IC1")
    ax.set_ylabel("IC2")
    ax.set_zlabel("IC3")
    ax.set_title("ICA on the Phishing data set")
    plt.show()
    print("-----------")
    x,y = newtx.shape
    for i in range(0, y):
        print(kurtosis(newtx[:,i], fisher=False))
    em(newtx, ty, newrx, ry, add="wICAtr", times=21, dataset=dataset, alg="ICA")
    em(newtx, ty, newrx, ry, ICA(n_components=2).fit_transform(tx), add="wICAtr", times=9, dataset=dataset, alg = "Ica")
    nn(newtx, ty, newrx, ry, add="wICAtr")
    myNN(newtx, ty, newrx, ry, "ica")

def myNN(tx,ty, rx, ry, alg=""):
    train_size = len(tx)
    offsets = range(int(0.1 * train_size), int(train_size), int(0.1 * train_size))
    train_err = [0] * len(offsets)
    test_err = [0] * len(offsets)
    cv_scores = [0] * len(offsets)
    for i, o in enumerate(offsets):
        print(o)
        layers = []
        for _ in range(5): layers.append(10)
        X_train_temp = X_train[:o].copy()
        y_train_temp = y_train[:o].copy()
        X_test_temp = X_test[:o].copy()
        y_test_temp = y_test[:o].copy()
        mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
           learning_rate_init=0.001, max_iter=220, momentum=0.9,nesterovs_momentum=True, random_state=1,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
        mlp.fit(X_train_temp, y_train_temp)
        train_err[i] = accuracy_score(y_train_temp, mlp.predict(X_train_temp))
        YpredTest = mlp.predict(X_test_temp)
        test_err[i] = accuracy_score(y_test_temp, YpredTest)
        print("test: "+str(test_err[i]))
        # test_err[i] = accuracy_score(y_test_temp, mlp.predict(X_test_temp))
    plt.figure()
    plt.plot(offsets, test_err, '-', label='test')
    plt.plot(offsets, train_err, '-', label='train')
    # plt.plot(offsets, cv_scores, '-', label='cross val')
    plt.legend()
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    title = 'Phishing NN: Accuracy vs Training Size - ' + alg
    plt.title(title)
    filename = 'Phishing_NN_TrainSize_'+alg+'.png'
    plt.savefig(filename, dpi=300)
    print 'plot complete'

    r = range(1,20)
    train_err = [0] * len(r)
    test_err = [0] * len(r)
    cv_scores = [0] * len(r)
    for l in r:
        print(l)
        layers = []
        for _ in range(l): layers.append(17)
        mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,nesterovs_momentum=True, random_state=1,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
        mlp.fit(X_train,y_train)
        train_err[l-1] = accuracy_score(y_train, mlp.predict(X_train))
        YpredTest = mlp.predict(X_test)
        test_err[l-1] = accuracy_score(y_test, YpredTest)

        # test_err[l-1] = accuracy_score(y_test, mlp.predict(X_test))
    plt.figure()
    plt.plot(r, test_err, '-', label='test')
    plt.plot(r, train_err, '-', label='train')
    # plt.plot(r, cv_scores, '-', label='cross val')
    plt.legend()
    plt.xlabel('Layers')
    plt.ylabel('Accuracy')
    title = 'Phishing NN: Performance vs Layers - ' + alg
    plt.title(title)
    filename = 'Phishing_NN_Layers'
    plt.savefig(filename+"_"+alg+".png", dpi=300)
    print 'plot complete'

    # second exp
    offsets = range(int(0.1 * 600), int(600), int(20))
    train_err = [0] * len(offsets)
    test_err = [0] * len(offsets)
    cv_scores = [0] * len(offsets)
    for i, o in enumerate(offsets):
        print(o)
        layers = []
        for _ in range(2): layers.append(14)
        mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
           learning_rate_init=0.001, max_iter=o, momentum=0.9,nesterovs_momentum=True, random_state=1,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
        tx = tx.reshape(-1,1)
        ry = ry.reshape(-1,1)
        print(ry.shape)
        # tx.reshape((7738,1))
        mlp.fit(tx,ty)
        train_err[i] = accuracy_score(ty, mlp.predict(tx))
        rx = rx.reshape(-1,1)
        YpredTest = mlp.predict(rx)
        test_err[i] = accuracy_score(ry, YpredTest)
        # test_err[i] = accuracy_score(y_test, mlp.predict(X_test))
    plt.figure()
    plt.plot(offsets, test_err, '-', label='test')
    plt.plot(offsets, train_err, '-', label='train')
    plt.legend()
    plt.xlabel('Max Iterations')
    plt.ylabel('Accuracy')
    title = 'Phishing NN: Accuracy vs Max Iterations'
    plt.title(title)
    filename = 'Phishing_NN_iter_' + alg+'.png'
    plt.savefig(filename, dpi=300)
    print 'plot complete'
    # print 'training_set_max_size:', train_size, '\n'
def kmtable(tx, ty, rx, ry, dataset=""):
    processed = []
    adj_rand = []
    v_meas = []
    mutual_info = []
    adj_mutual_info = []
    sil = []
    inertia = []


    compressor = PCA(n_components = tx[1].size/2)
    compressor.fit(tx, y=ty)
    pcatx = compressor.transform(tx)
    pcarx = compressor.transform(rx)
    p = []

    compressor = ICA(n_components = tx[1].size/2)
    compressor.fit(tx, y=ty)
    icatx = compressor.transform(tx)
    icarx = compressor.transform(rx)
    ic = []

    compressor = RandomProjection(tx[1].size/2)
    compressor.fit(tx, y=ty)
    rptx = compressor.transform(tx)
    rprx = compressor.transform(rx)
    r = []

    compressor = best(k=tx[1].size/2)
    compressor.fit(tx, y=ty)
    kbtx = compressor.transform(tx)
    kbrx = compressor.transform(rx)
    k = []
    for i in range(2,8):
        # clusters = {x:[] for x in range(i)}
        clf = KM(n_clusters=i)
        clf.fit(pcatx)
        test = clf.predict(pcatx)
        result = clf.predict(pcarx)
        p.append(metrics.v_measure_score(ry.ravel(), result))

        clf = KM(n_clusters=i)
        clf.fit(icatx)
        test = clf.predict(icatx)
        result = clf.predict(icarx)
        ic.append(metrics.v_measure_score(ry.ravel(), result))

        clf = KM(n_clusters=i)
        clf.fit(rptx)
        test = clf.predict(rptx)
        result = clf.predict(rprx)
        r.append(metrics.v_measure_score(ry.ravel(), result))

        clf = KM(n_clusters=i)
        clf.fit(kbtx)
        test = clf.predict(kbtx)
        result = clf.predict(kbrx)
        k.append(metrics.v_measure_score(ry.ravel(), result))
        # adj_rand.append(metrics.adjusted_rand_score(ry.ravel(), result))
        # v_meas.append(metrics.v_measure_score(ry.ravel(), result))
        # mutual_info.append(metrics.fowlkes_mallows_score(ry.ravel(), result))
        # adj_mutual_info.append(metrics.homogeneity_score(ry.ravel(), result))
    plt.figure()
    plt.title(dataset+": KM Clustering & DR")
    plt.xlabel('Number of clusters')
    plt.ylabel('V Measure Score value')
    plt.plot(range(2,8), p, label="PCA")
    plt.plot(range(2,8), ic, label="ICA")
    plt.plot(range(2,8), r, label = "RP")
    plt.plot(range(2,8), k, label="KB")
    plt.legend()
    plt.ylim(ymin=-0.05, ymax=0.5)
    plt.savefig("KM_DR_"+dataset+"_VM.png", dpi=300)

def graphCallerNN(tx, ty, rx, ry):
    n = tx[1].size/2
    compressor = PCA(n_components = n)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    newtx = oneem(newtx, ty, newrx, ry)
    myNN(newtx, ty, newrx, ry, "EM-PCA")
    # nnTable(newtx, ty, newrx, ry, alg="EM-PCA")

    compressor = ICA(n_components = n)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    newtx = oneem(newtx, ty, newrx, ry)
    nnTable(newtx, ty, newrx, ry, alg="EM-ICA")
    myNN(newtx, ty, newrx, ry, "EM-Ica")

    compressor = RandomProjection(n)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    newtx = oneem(newtx, ty, newrx, ry)
    nnTable(newtx, ty, newrx, ry, alg="EM-RP")
    myNN(newtx, ty, newrx, ry, "EM-RP")

    compressor = best(k=n)
    compressor.fit(tx, y=ty)
    newtx = compressor.transform(tx)
    newrx = compressor.transform(rx)
    newtx = oneem(newtx, ty, newrx, ry)
    nnTable(newtx, ty, newrx, ry, alg="EM-KB")
    myNN(newtx, ty, newrx, ry, "EM-KB")

def oneem(tx, ty, rx, ry, add="", times=5):
    scores = []
    clf = EM(n_components=times)
    clf.fit(tx)  #fit it to our data
    scores.append(clf.predict_proba(tx))
    scores.append(clf.predict_proba(rx))
    return scores

def nnTable(tx, ty, rx, ry, add="", iterations=150, dataset="", alg=""):
    tx = nptx.reshape(-1,1)
    layers = []
    for _ in range(2): layers.append(14)
    mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(layers), learning_rate='constant',
       learning_rate_init=0.001, max_iter=380, momentum=0.9,nesterovs_momentum=True, random_state=1,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1)
    start = time.time()
    mlp.fit(tx,ty)
    end = time.time()-start
    print(end)
    print("train: " + str(accuracy_score(ty, mlp.predict(tx))))
    YpredTest = mlp.predict(rx)
    print("test: " + str(accuracy_score(ry, YpredTest)))

def caller(tx, ty, rx, ry):
    nums = [4,8,12,16]
    for n in nums:
        print("PCA")
        print(n)
        compressor = PCA(n_components = n)
        compressor.fit(tx, y=ty)
        newtx = compressor.transform(tx)
        newrx = compressor.transform(rx)
        nnTable(newtx, ty, newrx, ry, alg="PCA")
    for n in nums:
        print("ICA")
        print(n)
        compressor = ICA(n_components = n)
        compressor.fit(tx, y=ty)
        newtx = compressor.transform(tx)
        newrx = compressor.transform(rx)
        nnTable(newtx, ty, newrx, ry, alg="ICA")
    for n in nums:
        print("RandProj")
        print(n)
        compressor = RandomProjection(n)
        compressor.fit(tx, y=ty)
        newtx = compressor.transform(tx)
        newrx = compressor.transform(rx)
        nnTable(newtx, ty, newrx, ry, alg="PCA")
    for n in nums:
        print("kbest")
        print(n)
        compressor = best(k=n)
        compressor.fit(tx, y=ty)
        newtx = compressor.transform(tx)
        newrx = compressor.transform(rx)
        nnTable(newtx, ty, newrx, ry, alg="PCA")

def nn(tx, ty, rx, ry, add="", iterations=250, dataset=""):
    """
    trains and plots a neural network on the data we have
    """
    resultst = []
    resultsr = []
    positions = range(iterations)
    network = buildNetwork(tx[1].size, 5, 1, bias=True)
    ds = ClassificationDataSet(tx[1].size, 1)
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.01)
    train = zip(tx, ty)
    test = zip(rx, ry)
    for i in positions:
        trainer.train()
        resultst.append(sum(np.array([(round(network.activate(t_x)) - t_y)**2 for t_x, t_y in train])/float(len(train))))
        resultsr.append(sum(np.array([(round(network.activate(t_x)) - t_y)**2 for t_x, t_y in test])/float(len(test))))
        # resultsr.append(sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry)))
        print i, resultst[-1], resultsr[-1]
    plot([0, iterations, 0, 1], (positions, resultst, "ro", positions, resultsr, "bo"), "Network Epoch", "Percent Error", "Neural Network Error", "NN"+add)


def plot(axes, values, x_label, y_label, title, name):
    plt.clf()
    plt.plot(*values)
    plt.axis(axes)
    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(-10,40)
    plt.xlabel(x_label)
    plt.savefig(name+".png", dpi=500)
    # plt.show()
    plt.clf()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

if __name__=="__main__":
    df = pd.read_csv("car_evaluation.csv", header=0, sep = ",", quotechar = '"')
    df = df.replace('vhigh', 4)
    df = df.replace('high', 3)
    df = df.replace('med', 2)
    df = df.replace('low', 1)
    df = df.replace('5more', 6)
    df = df.replace('more', 5)
    df = df.replace('small', 1)
    df = df.replace('big', 3)
    df = df.replace('unacc', 1)
    df = df.replace('acc', 2)
    df = df.replace('good', 3)
    df = df.replace('vgood', 4)

    car = df.values
    X = np.array(df.drop(['class'], 1).astype(int))
    y = np.array(df['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
    dataset = "Car"
    kmtable(X_train, y_train, X_test, y_test,dataset=dataset)
    km(X_train, y_train, X_test, y_test, [], times = 35, dataset=dataset); print 'km done'
    kbest(X_train, y_train, X_test, y_test, dataset); print 'kbest done'
    pca(X_train, y_train, X_test, y_test, dataset); print 'pca 1 done'
    ica(X_train, y_train, X_test, y_test, dataset); print 'ica done'
    randproj(X_train, y_train, X_test, y_test, dataset); print 'randproj done'

    df = pd.read_csv("dataset.csv", sep = ",", quotechar = '"')
    df = pd.get_dummies(df)
    X = np.array(df.drop(['Result'], 1).astype(int))
    y = np.array(df['Result'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
    dataset = "Phishing"

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    kmtable(X_train, y_train, X_test, y_test,dataset=dataset)
    kbest(X_train, y_train, X_test, y_test, dataset); print 'kbest done'
    km(X_train, y_train, X_test, y_test, [], times = 35, dataset=dataset); print 'em done'
    caller(X_train, y_train, X_test, y_test)
    graphCallerNN(X_train, y_train, X_test, y_test)
    pca(X_train, y_train, X_test, y_test, dataset); print 'pca 2 done'
    ica(X_train, y_train, X_test, y_test, dataset); print 'ica done'
    randproj(X_train, y_train, X_test, y_test, dataset); print 'randproj done'
    kbest(X_train, y_train, X_test, y_test, dataset); print 'kbest done'
