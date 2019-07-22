#%% Start a fig

# Select an Image
# image_id = df_by_image.index[-1]
# image_id = df_by_image.index[9] # Select an image with 15 ships
image_id = np.random.choice(df[df['HasShip']].index.values)

img, contours = get_ellipsed_images(image_id)
plt.figure()
plt.interactive(True)
plt.interactive(False)
plt.imshow(img)

#%%
logging.info("Processing image of shape {}".format(img.shape))
data = img / 255
logging.info("Changed values to 0-1 range".format(img.shape))
data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
logging.info("Reshape to pixel list {}".format(data.shape))

#%%
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    ax[2].scatter(G, B, color=colors, marker='.')
    ax[2].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);


def plot_pixels3D(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(R,G,B, color=colors, marker='.', depthshade=False)
    ax.set(xlabel='Red', ylabel='Green', zlabel='Blue', xlim=(0, 1), ylim=(0, 1))

    # cur_axes = plt.gca()
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.get_zaxis().set_ticklabels([])
    # axis.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False

    # Get rid of the ticks
    # axis.set_xticks([])
    # axis.set_yticks([])
    # axis.set_zticks([])
    fig.suptitle(title, size=20)

#%%

# plot_pixels(data, title='Input color space: 16 million possible colors')
# plt.show()

#%%
plot_pixels3D(data, title='Input color space: 16 million possible colors')
plt.show()

#%% Run the K Means

import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans
num_clusters = 2
kmeans = MiniBatchKMeans(2)
kmeans.fit(data)

all_new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

all_cluster_labels = kmeans.labels_
cluster_counts = np.bincount(all_cluster_labels).tolist()
cluster_names = np.unique(all_cluster_labels).tolist()
cluster_colors = np.unique(all_new_colors, axis=0)
# cluster_colors *= 255

N_points = 20000

# Generate a list of 20000 indices
np.random.RandomState(0) # Init the RNG
i = rng.permutation(data.shape[0])[:N_points]
colors_i = all_new_colors[i]
labels_i = all_cluster_labels[i]
R, G, B = data[i].T

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

# cluster_markers = ['x','+']
cluster_markers = ['1','+']

for cluster_name, cluster_marker in zip(cluster_names, cluster_markers):
    print(cluster_name, cluster_marker)
    this_cluster_mask = labels_i == cluster_name

    logging.info("Cluster {} with {} points".format(cluster_name, sum(this_cluster_mask)))

    # sum(this_cluster_mask)

    ax.scatter(R[this_cluster_mask], G[this_cluster_mask], B[this_cluster_mask], color=colors_i[this_cluster_mask], marker=cluster_marker, depthshade=False)

ax.set(xlabel='Red', ylabel='Green', zlabel='Blue', xlim=(0, 1), ylim=(0, 1))
plt.show()


plt.show()

print(list(zip(cluster_names, cluster_counts)))

plot_pixels3D(data, colors=all_new_colors,
              title="Reduced color space: {} colors".format(num_clusters))

clusters = np.zeros(all_new_colors.shape[0], dtype=int)

all_new_colors.mean()


plt.show()



#%%














#%%

img_small = cv2.resize(img, (200,200))
plt.imshow(img_small)
plt.show()
r, g, b = cv2.split(img_small)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

# fig = plt.figure()
# ax = fig.gca(projection='3d')

image_small_hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
data = img_small
data = image_small_hsv
data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

# pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))

norm = colors_i.Normalize(vmin=-1., vmax=1.)
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



#%% OLD

from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors_i

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
