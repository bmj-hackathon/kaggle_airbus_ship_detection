from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()


from sklearn.cluster import KMeans
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
#%%
img.shape

data = img / 255
data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
data.shape


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);

plot_pixels(data, title='Input color space: 16 million possible colors')
plt.show()
#%%

img_small = cv2.resize(img, (200,200))
plt.imshow(img_small)
plt.show()
r, g, b = cv2.split(img_small)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")


image_small_hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
data = img_small
data = image_small_hsv
data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

# pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))

norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(data)
data = norm(data).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=data, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")

plt.show()
#%%
plt.imshow(img)
plt.show()
import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans
num_clusters = 2
kmeans = MiniBatchKMeans(2)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors,
            title="Reduced color space: 16 colors")

plt.show()
