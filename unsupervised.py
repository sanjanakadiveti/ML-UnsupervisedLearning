from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition.pca import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import preprocessing
import sklearn.utils as utils

colors = ['r', 'g', 'b', 'y', 'c', 'm','#eeefff', '#317c15', '#4479b4', '#6b2b9c',
'#63133b', '#6c0d22', '#0c7c8c', '#67c50e','#c5670e', '#946c47', '#58902a', '#54b4e4',
'#e4549e', '#2b2e85'  ]

def plot_results(title, x_val, rand_score, v_score, mutual_score, x_label, y_label):
    plt.figure(1)
    plt.plot(x_val, rand_score, 'o-', color='b', label='adj_rand_score')
    plt.plot(x_val, v_score, 'o-', color='r', label='v_measure_score')
    plt.plot(x_val, mutual_score, 'o-', color='g', label='adj_mutual_score')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def pca(X, y, components, max_cluster, num_classes, run_nn=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
        y, test_size=0.3, train_size=0.7, shuffle=True)
    pca_compress = PCA(n_components=components, whiten=True)
    pca_compress.fit(X_train, y=y_train)
    X_train_new = pca_compress.transform(X_train)
    X_test_new = pca_compress.transform(X_test)
    X_original = pca_compress.inverse_transform(X_test_new)
    loss = ((X_test - X_original)**2).mean()
    print("Reconstruction Error " + str(loss))
    eigenvalues = pca_compress.explained_variance_
    print(eigenvalues)
    if run_nn:
        mlp_classifier(X_train_new, y_train, 0.3, plot=True, X_test=X_test_new, y_test=y_test)
    X_new = np.concatenate((X_train_new, X_test_new), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    kmeans(X_new, y,max_cluster, num_classes, run_nn=run_nn, plot_cluster=True,
        reduction_algo='PCA')
    expectation_max(X_new, y, max_cluster, num_classes, run_nn=run_nn,
        plot_cluster=True, reduction_algo='PCA')
def ica(X, y, components, max_cluster, num_classes, run_nn=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
        y, test_size=0.3, train_size=0.7, shuffle=True)
    ica_compress = FastICA(n_components=components, whiten=True)
    ica_compress.fit(X_train, y=y_train)
    X_train_new = ica_compress.transform(X_train)
    X_test_new = ica_compress.transform(X_test)
    print(kurtosis(X_test_new))
    print(ica_compress.components_)
    if run_nn:
        mlp_classifier(X_train_new, y_train, 0.3, plot=True,
            X_test=X_test_new, y_test=y_test)
    X_new = np.concatenate((X_train_new, X_test_new), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    kmeans(X_new, y,max_cluster, num_classes, run_nn=run_nn, plot_cluster=True,
        reduction_algo='ICA')
    expectation_max(X_new, y, max_cluster, num_classes, run_nn=run_nn,
        plot_cluster=True, reduction_algo='ICA')
def random_projection(X, y, components, max_cluster, num_classes, run_nn=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
        y, test_size=0.3, train_size=0.7, shuffle=True)
    random_proj = GaussianRandomProjection(n_components=components)
    random_proj.fit(X_train, y=y_train)
    print(random_proj.components_)
    X_train_new = random_proj.transform(X_train)
    X_test_new = random_proj.transform(X_test)
    inverse_components = np.linalg.pinv(random_proj.components_)
    reconstructed_instances = utils.extmath.safe_sparse_dot(X_test_new, inverse_components.T)
    loss = ((X_test - reconstructed_instances)**2).mean()
    print("Reconstruction Error " + str(loss))
    if run_nn:
        mlp_classifier(X_train_new, y_train, 0.3, plot=True, X_test=X_test_new,
            y_test=y_test)
    X_new = np.concatenate((X_train_new, X_test_new), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    kmeans(X_new, y,max_cluster, num_classes, run_nn=run_nn, plot_cluster=True,
        reduction_algo='Random Projection')
    expectation_max(X_new, y, max_cluster, num_classes, run_nn=run_nn,
        plot_cluster=True, reduction_algo='Random Projection')

def lda(X, y, components, max_cluster, num_classes, run_nn=False, plot_cluster=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
        y, test_size=0.3, train_size=0.7, shuffle=True)
    linear_disc = LinearDiscriminantAnalysis(solver='svd', n_components=components)
    linear_disc.fit(X_train, y_train)
    X_train_new = linear_disc.transform(X_train)
    X_test_new = linear_disc.transform(X_test)
    if run_nn:
       mlp_classifier(X_train_new, y_train, 0.3, plot=True, X_test=X_test_new,
        y_test=y_test)
    X_new = np.concatenate((X_train_new, X_test_new), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    kmeans(X_new, y,max_cluster, num_classes, run_nn=run_nn, plot_cluster=plot_cluster,
        reduction_algo='LDA')
    expectation_max(X_new, y, max_cluster, num_classes, run_nn=run_nn,
        plot_cluster=plot_cluster, reduction_algo='LDA')

def kmeans(X, y, max_cluster, num_classes, run_nn=False, plot_cluster=False,
          reduction_algo=''):
    adj_rand_score = []
    adj_mutual_score = []
    v_measure_score = []
    best_cluster = 2
    best_score = -10000
    for i in range(2, max_cluster):
        clusterer = KMeans(n_clusters = i, max_iter=1000, n_jobs=1)
        clusterer.fit(X)
        predicted_labels = clusterer.predict(X)
        score = metrics.adjusted_rand_score(y, predicted_labels)
        mutual_score = metrics.adjusted_mutual_info_score(y, predicted_labels)
        v_score = metrics.v_measure_score(y, predicted_labels)
        adj_rand_score.append(score)
        adj_mutual_score.append(mutual_score)
        v_measure_score.append(v_score)
        if v_score > best_score:
            best_score = v_score
            best_cluster = i
    plot_results("KMeans Score " + reduction_algo, range(2, max_cluster),
        adj_rand_score, adj_mutual_score, v_measure_score, "Number of Clusters", "Score")
    if plot_cluster:
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        best_clusterer = KMeans(n_clusters=best_cluster)
        best_clusterer.fit(X)
        Z = best_clusterer.predict(X)
        print(len(Z))
        print(len(X))
        plt.figure(1)
        plt.clf()

        for i in range(0, len(X)):
            plt.plot(X[i][0], X[i][1], marker='.', color=colors[Z[i]], markersize=2)


        #plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = best_clusterer.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='k', zorder=10)
        plt.title('K-means Clusters with ' + reduction_algo + '\n'
                  'Centroids are marked with black cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()



    if run_nn:
        best_clusterer = KMeans(n_clusters=best_cluster, max_iter=1000, n_jobs=1)
        best_clusterer.fit(X)
        output = best_clusterer.predict(X)
        new_feature = np.reshape(output, (output.size, 1))
        X = np.append(X, new_feature, 1)
        mlp_classifier(X, y, 0.3, plot=True)

def expectation_max(X, y, max_component, num_classes, run_nn=False,
    plot_cluster=False, reduction_algo=''):
    adj_rand_score = []
    adj_mutual_score = []
    v_measure_score = []
    best_cluster = 2
    best_score = -10000
    for i in range(2, max_component):
        clusterer = GaussianMixture(n_components = i, max_iter=1000)
        clusterer.fit(X)
        predicted_labels = clusterer.predict(X)
        score = metrics.adjusted_rand_score(y, predicted_labels)
        mutual_score = metrics.adjusted_mutual_info_score(y, predicted_labels)
        v_score = metrics.v_measure_score(y, predicted_labels)
        adj_rand_score.append(score)
        adj_mutual_score.append(mutual_score)
        v_measure_score.append(v_score)
        if v_score > best_score:
            best_score = v_score
            best_cluster = i
    plot_results("EM Score " + reduction_algo, range(2, max_component),
        adj_rand_score, adj_mutual_score, v_measure_score, "Number of Clusters", "Score")
    if plot_cluster:
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        best_clusterer = GaussianMixture(n_components=best_cluster, max_iter=1000)
        best_clusterer.fit(X)
        Z = best_clusterer.predict(X)
        print(len(Z))
        print(len(X))
        plt.figure(1)
        plt.clf()
        for i in range(0, len(X)):
            plt.plot(X[i][0], X[i][1], marker='.', color=colors[Z[i]], markersize=2)
        # Plot the centroids as a white X
        centroids = best_clusterer.means_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='k', zorder=10)
        plt.title('EM Clusters with ' + reduction_algo +'\n'
                  'Centroids are marked with black cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


    if run_nn:
        best_clusterer = GaussianMixture(n_components=best_cluster, max_iter=1000)
        best_clusterer.fit(X)
        output = best_clusterer.predict(X)
        new_feature = np.reshape(output, (output.size, 1))
        X = np.append(X, new_feature, 1)
        mlp_classifier(X, y, 0.3, plot=True)

def mlp_classifier(X, y, split_amount, plot=True, X_test=None, y_test=None):
    '''
    X - inputs
    y- target values
    split_amount - percentage of dataset to be set aside for testing
    '''
    training_size = 1 - split_amount
    scaler = StandardScaler()
    X_train = None
    y_train = None
    if X_test is None and y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_amount, train_size=training_size, shuffle=True)
    else:
        X_train = X
        y_train = y
    neural_net = MLPClassifier(hidden_layer_sizes=(6,4,3,2), activation='logistic', learning_rate='constant', solver='lbfgs')

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    max_iter_range = range(100, 1500, 100)

    train_scores, test_scores = validation_curve(neural_net, X_train, y_train, param_name='max_iter'
    , param_range=max_iter_range, cv=5, n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    best_iter = max_iter_range[list(test_scores_mean).index(max(test_scores_mean))]


    neural_net.set_params(max_iter=best_iter)
    training_sizes = np.linspace(.1, 1.0, 5)
    train_sizes, train_scores_learn, test_scores_learn = learning_curve(neural_net, X_train, y_train,
        train_sizes=training_sizes, cv=5, n_jobs=1)


    train_scores_learn_mean = np.mean(train_scores_learn, axis=1)
    train_scores_learn_std = np.std(train_scores_learn, axis=1)
    test_scores_learn_mean = np.mean(test_scores_learn, axis=1)
    test_scores_learn_std = np.std(test_scores_learn, axis=1)

    neural_net.fit(X_train, y_train)
    measure_performance(X_test, y_test, neural_net)
    if plot:
        lw=2
        plt.figure()
        plt.grid()
        plt.title("Neural Network Validation Curve")
        plt.plot(max_iter_range, train_scores_mean, label='training_score', color='darkorange')
        plt.fill_between(max_iter_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
        plt.plot(max_iter_range, test_scores_mean, label='cross_validation_score', color='navy')
        plt.fill_between(max_iter_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Score')
        title = "Neural Network Learning Curve (Max Iterations = " + str(best_iter) + " )"
        plt.figure(2)
        plt.grid()
        plt.title(title)
        plt.fill_between(train_sizes, train_scores_learn_mean - train_scores_learn_std,
                         train_scores_learn_mean + train_scores_learn_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_learn_mean  - test_scores_learn_std,
                         test_scores_learn_mean + test_scores_learn_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_learn_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_learn_mean, 'o-', color="g",
                 label="Test score")
        plt.xlabel('Training Sizes')
        plt.ylabel('Score')
        plt.legend()
        plt.show()


def measure_performance(X, y, neural_net):
    y_predict = neural_net.predict(X)
    print("\n\nTest Accuracy: " + str(metrics.accuracy_score(y, y_predict, normalize=True)) + "\n\n")

    print("\n\nClassification report:\n\n")
    print(metrics.classification_report(y, y_predict))

    print("\n\nConfusion Matrix:\n\n")
    print(metrics.confusion_matrix(y, y_predict))


# if sys.argv[1] == 'student':
#     df = preprocessing.combine_csv("data/student-mat.csv", "data/student-por.csv")
#     all_classes = list(df.columns)
#     df_string = list(df.select_dtypes(include=['object']).columns)
#     df_encoded = preprocessing.encode_features(df_string, df, reduce_classes=True)
#     X,y = preprocessing.create_inputs(df_encoded, "G3")
#     if sys.argv[2] == 'nn':
#         mlp_classifier(X, y, 0.3, plot=True)
#     elif sys.argv[2] == 'kmeans':
#         kmeans(X, y, 20, 4, run_nn=False, plot_cluster=False)
#     elif sys.argv[2] == 'EM':
#         expectation_max(X, y, 20, 4, run_nn=False, plot_cluster=False)
#     elif sys.argv[2] =='PCA':
#         pca(X, y, 4, 20, 4, run_nn=True)
#     elif sys.argv[2] == 'ICA':
#         ica(X, y, 4, 20, 4, run_nn=True)
#     elif sys.argv[2] == 'randproj':
#         random_projection(X, y, 4, 20, 4, run_nn=True)
#     elif sys.argv[2] == 'LDA':
#         lda(X, y, 4, 20, 4, run_nn=True, plot_cluster=True)
# if sys.argv[1] == 'wine':
#     dfwine = preprocessing.get_csv_data("data/winequality_white.csv", index=None)
#     dfwine = preprocessing.wine_binary_classes(dfwine)
#     X,y = preprocessing.create_inputs(dfwine, "quality", num_features=11)
#     if sys.argv[2] == 'nn':
#         mlp_classifier(X_train, y_train, 0.0, plot=True,
#         X_test=X_test, y_test=y_test)
#     elif sys.argv[2] == 'kmeans':
#         kmeans(X, y, 20, 2, run_nn=False, plot_cluster=False)
#     elif sys.argv[2] == 'EM':
#         expectation_max(X, y, 20, 2, run_nn=False, plot_cluster=False)
#     elif sys.argv[2] == 'PCA':
#         pca(X, y, 2, 20, 2)
#     elif sys.argv[2] == 'ICA':
#         ica(X, y, 2, 20, 2)
#     elif sys.argv[2] == 'randproj':
#         random_projection(X, y, 2, 20, 2)
#     elif sys.argv[2] == 'LDA':
#         lda(X, y, 3, 20, 2)

var = 'km'
df = preprocessing.get_csv_data("dataset.csv")
X,y = preprocessing.create_inputs(df, "Result")
# ica(X, y, 16, 4, 2, run_nn=False)
random_projection(X, y, 16, 4, 2, run_nn=True)
# all_classes = list(df.columns)
# df_string = list(df.select_dtypes(include=['object']).columns)
# df_encoded = preprocessing.encode_features(df_string, df, reduce_classes=True)

if var == 'nn':
    mlp_classifier(X, y, 0.3, plot=True)
elif var == 'kmeans':
    kmeans(X, y, 4, 2, run_nn=False, plot_cluster=False)
elif var == 'EM':
    expectation_max(X, y, 20, 4, run_nn=False, plot_cluster=False)
elif var =='PCA':
    pca(X, y, 4, 20, 4, run_nn=True)
elif var == 'ICA':
    ica(X, y, 4, 20, 4, run_nn=True)
elif var == 'randproj':
    random_projection(X, y, 4, 20, 4, run_nn=True)
elif var == 'LDA':
    lda(X, y, 4, 20, 4, run_nn=True, plot_cluster=True)

df = preprocessing.get_csv_data("car_evaluation.csv")
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
X,y = preprocessing.create_inputs(df, "class")
# ica(X, y, 4, 8, 4, run_nn=False)
random_projection(X, y, 4, 8, 4, run_nn=True)
