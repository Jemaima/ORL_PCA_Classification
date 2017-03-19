from sklearn import linear_model
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import numpy as np


def import_face_data():
    data = fetch_olivetti_faces(shuffle=True, random_state=rdm)
    return data


def plot_images(title, images):
    plt.figure()
    plt.suptitle(title, size=16)
    for i in range(n1*n2):
        plt.subplot(plt.subplot(n1, n2, i + 1))
        plt.imshow(np.reshape(images[i], img_size), cmap='gray')
        plt.axis("off")  # removes the axis and the ticks


def plot_pca_params(local_pca):
    # plot explained variance
    plt.figure()
    plt.xlim([0, 100])
    plt.plot(local_pca.explained_variance_, linewidth=2)
    # plot mean face
    plt.figure()
    plt.suptitle('Mean Face', size=14)
    plt.axis("off")
    plt.imshow(np.reshape(local_pca.mean_, img_size), cmap='gray')


def pca_decomposition(n_el, plot=False):
    p = PCA(n_components=n_el)
    p.fit(x_train)
    reduced_images = p.transform(x_train)
    inverse_trans_images = p.inverse_transform(reduced_images)
    if plot:
        plot_pca_params(p)
        plot_images('Reconstructed faces', inverse_trans_images[:n1 * n2])
    return p


def compute_plot_roc(y):
    target_score = np.zeros(shape=(len(y_test), 40))
    for i, row in enumerate(target_score):
        row[y_test[i]] = 1
    target_score = target_score.flatten()
    y = y.flatten()
    fpr, tpr, _ = roc_curve(target_score, y)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot([0, 1], [1, 1], color='red', lw=lw, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# Initial data
rdm = np.random.RandomState(1)
n1 = 2
n2 = 3
work_data = import_face_data()
data_l = len(work_data['data'])
img_size = work_data['images'][0].shape
x_train, x_test, y_train, y_test = train_test_split(work_data['data'],
                                                    work_data['target'],
                                                    test_size=0.2, random_state=rdm)
# Plot first images
plot_images('Initial faces', x_train[:n1*n2])
print('Data loaded.')

# OneVsRestClassifier
pca = pca_decomposition(20, True)
classifier = OneVsRestClassifier(linear_model.LinearRegression())
classifier.fit(pca.transform(x_train), y_train)
y_score = classifier.decision_function(pca.transform(x_test))
compute_plot_roc(y_score)
print('Finished')
