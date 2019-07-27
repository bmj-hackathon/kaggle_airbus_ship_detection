def summary_kmeans(kmeans):
    # all_new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

    all_cluster_labels = kmeans.labels_
    cluster_counts = np.bincount(all_cluster_labels).tolist()
    cluster_names = np.unique(all_cluster_labels).tolist()
    cluster_colors = np.unique(all_new_colors, axis=0)

def get_kmeans_color(img, _kmeans):
    """asdf

    :param ax: The axis to plot into
    :param img: The orignal image
    :param _kmeans: The FIT kmeans object
    :return:
    """

    original_pixels = img / 255
    new_shape = original_pixels.shape[0] * original_pixels.shape[1], original_pixels.shape[2]
    original_pixel_locations_flat = original_pixels.reshape(new_shape)

    new_pixel_colors = _kmeans.cluster_centers_[_kmeans.predict(original_pixel_locations_flat)]
    logging.info("New pixels, shape {}".format(new_pixel_colors.shape))
    logging.info("Colors: {}".format(np.unique(new_pixel_colors, axis=0)))

    cluster_labels = np.unique(_kmeans.labels_).tolist()
    logging.info("{} custers: {}".format(len(cluster_labels), cluster_labels))

    N_points = 20000
    # Generate a list of 20000 indices
    rng = np.random.RandomState(0)
    i = rng.permutation(original_pixel_locations_flat.shape[0])[:N_points]
    logging.info("Sampling {} points".format(len(i)))

    pixel_locations = original_pixel_locations_flat[i]
    logging.info("Returning pixel locations: {}".format(pixel_locations.shape))
    color_vec_i = new_pixel_colors[i]
    logging.info("Returning colors: {}".format(color_vec_i.shape))
    labels_vec_i = _kmeans.labels_[i]
    logging.info("Returning labels: {}".format(labels_vec_i))

    return pixel_locations, color_vec_i, labels_vec_i

def plot_kmeans_color2(pixel_locs, colors, labels):
    fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
    ax = plt.axes(projection="3d")
    R, G, B = pixel_locs.T

    for label in np.unique(labels).tolist():
        this_cluster_mask = labels == label
        ax.scatter(R[this_cluster_mask], G[this_cluster_mask], B[this_cluster_mask], color=colors[this_cluster_mask], depthshade=False)

    ax.set(xlabel='Red', ylabel='Green', zlabel='Blue', xlim=(0, 1), ylim=(0, 1))


# TODO: This works...
pixel_locs, colors, labels = get_kmeans_color(image.img, kmeans)
plot_kmeans_color2(pixel_locs, colors, labels)
plt.show()

#%%
fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
ax = plt.axes(projection="3d")
plot_kmeans_color(ax, image.img, kmeans)
plt.show()

#%%
image_id = df_by_image.index[-1] # Select an image with 15 ships
image_id = df_by_image.index[10] # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df)
image.load_ships()
r = image.records

image.ship_summary_table()
kmeans = image.k_means(num_clusters=2)

#%%
plt.imshow(image.img)
plt.show()

#%%

kmeans_image = fit_kmeans_pixels(image.img, kmeans)
plt.imshow(kmeans_image)
plt.show()

#%%

#%%
# TODO: This doesn't work??
fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
ax = plt.axes(projection="3d")

# Flatten to a list of pixels
new_shape = kmeans_image.shape[0] * kmeans_image.shape[1], kmeans_image.shape[2]
pixels_vector = kmeans_image.reshape(new_shape)
N_points = 20000

# Downsampling to a random list of indices
N_points = 20000
rng = np.random.RandomState(0)
i = rng.permutation(pixels_vector.shape[0])[:N_points]

# Sampled labels (cluster num)
cluster_numbers = np.unique(kmeans.labels_).tolist()
labels = kmeans.labels_[i]
# RGB vectors from the sampled pixels vector
# TODO: Do not split into RGB vectors!?
R, G, B = pixels_vector[i].T
# Cluster integer labels


# A list of markers
# TODO: Expand marker list for more clusters, select down, wrap end
cluster_markers = ['1', '+']

for cluster_num, cluster_marker in zip(cluster_numbers, cluster_markers):
    # print(cluster_num, cluster_marker)
    this_cluster_mask = labels == cluster_num
    # logging.info("Cluster {} with {} points".format(cluster_num, sum(this_cluster_mask)))
    # sum(this_cluster_mask)
    ax.scatter(R[this_cluster_mask], G[this_cluster_mask], B[this_cluster_mask], color=pixels_vector[i][this_cluster_mask],
               marker=cluster_marker)

ax.set(xlabel='Red', ylabel='Green', zlabel='Blue', xlim=(0, 1), ylim=(0, 1))
plt.show()

#%% OLD!!!

def plot_kmeans_color(ax, img, _kmeans):
    """asdf

    :param ax: The axis to plot into
    :param img: The orignal image
    :param _kmeans: The FIT kmeans object
    :return:
    """

    original_pixels = img / 255
    new_shape = original_pixels.shape[0] * original_pixels.shape[1], original_pixels.shape[2]
    original_pixel_locations_flat = original_pixels.reshape(new_shape)

    new_pixel_colors = _kmeans.cluster_centers_[_kmeans.predict(original_pixel_locations_flat)]
    logging.info("New pixels, shape {}".format(new_pixel_colors.shape))
    logging.info("Colors: {}".format(np.unique(new_pixel_colors, axis=0)))

    cluster_labels = np.unique(_kmeans.labels_).tolist()
    logging.info("{} custers: {}".format(len(cluster_labels), cluster_labels))

    N_points = 20000
    # Generate a list of 20000 indices
    rng = np.random.RandomState(0)
    i = rng.permutation(original_pixel_locations_flat.shape[0])[:N_points]
    logging.info("Sampling {} points".format(len(i)))


    pixel_locations_i = original_pixel_locations_flat[i].T
    colors_i = new_pixel_colors[i]
    labels_i = _kmeans.labels_[i]
    R, G, B = pixels_i

    cluster_markers = ['1','+']

    for cluster_name, cluster_marker in zip(cluster_labels, cluster_markers):
        print(cluster_name, cluster_marker)
        this_cluster_mask = labels_i == cluster_name

        logging.info("Cluster {} with {} points".format(cluster_name, sum(this_cluster_mask)))

        # sum(this_cluster_mask)

        ax.scatter(R[this_cluster_mask], G[this_cluster_mask], B[this_cluster_mask], color=colors_i[this_cluster_mask], marker=cluster_marker, depthshade=False)

    ax.set(xlabel='Red', ylabel='Green', zlabel='Blue', xlim=(0, 1), ylim=(0, 1))


#%%
# OLD
summary_kmeans(kmeans)

kmeans.cluster_centers_

# rotated_rect

# %% Start a fig
# OLD

# Select an Image
# image_id = df_by_image.index[-1]
image_id = df_by_image.index[9] # Select an image with 15 ships
image_id = df_by_image.index[1] # Select an image with 15 ships
image_id = df_by_image.index[2] # Select an image with 15 ships
# image_id = np.random.choice(df[df['HasShip']].index.values)

plt.interactive(True)
plt.interactive(False)

#%%
# def plot_summary(image_id):



#%%
#### Create fig ####
fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
#### Define the layout ####
gs = plt.GridSpec(3, 3)
# Adjust margins
gs.update(left=0.1, top=0.90)

# Name and palce the axes
ax_image = plt.subplot(gs[0:2, 0:2])
ax_hist = plt.subplot(gs[-1, 0:2])
ax_1 = plt.subplot(gs[0, 2], projection="3d")
ax_2 = plt.subplot(gs[1, 2])
ax_3 = plt.subplot(gs[2, 2])

# fig.add_subplot(1, 1, 1,

# Manually shift the image axis
pos_image1 = ax_image.get_position()
pos_image2 = [pos_image1.x0 + 0.1, pos_image1.y0, pos_image1.width * 0.9, pos_image1.height]
ax_image.set_position(pos_image2)

# Text summary: Place a new axis object
pos_text = [pos_image1.x0 - 0.05, pos_image1.y0, pos_image1.width * 0.35, pos_image1.height]
ax_text = plt.axes(pos_text)
# ax_text.get_xaxis().set_visible(False)
# ax_text.get_yaxis().set_visible(False)
ax_text.axis('off')

#### Title ####
title_string = "{}, {} ships".format(image_id, int(df_by_image.loc[image_id, 'TotalShips']))
fig.suptitle(title_string, x=0.05, ha='left')

#### Summary table ####
# table_df = pd.DataFrame(columns=['#', 'x', 'y', 'angle', 'area')
object_summary = list()
img, contours = get_ellipsed_images(image_id)
for i, c in enumerate(contours):
    M = cv2.moments(c)
    this_ship = dict()
    this_ship['#'] = i
    this_ship['x'] = round(M['m10'] / M['m00'])
    this_ship['y'] = round(M['m01'] / M['m00'])
    this_ship['area'] = int(cv2.contourArea(c))
    rotated_rect = cv2.fitEllipse(c)
    this_ship['angle'] = round(rotated_rect[2])

    object_summary.append(this_ship)

table_df = pd.DataFrame(object_summary)

# ax_text.text( summary_string,verticalalignment='top')
# mpl.table.Table(ax_text, loc)
table_obj = ax_text.table(cellText=table_df.values, colLabels=table_df.columns, loc='upper left')
for (row, col), cell in table_obj.get_celld().items():
  if (row == 0) or (col == -1):
    cell.set_text_props(fontproperties=mpl.font_manager.FontProperties(weight='bold'))


#### Plot image ####
ax_image.imshow(img)
ax_image.axis('off')

#### Plot histogram ####
plot_hist(img, ax_hist)

#### Plot kmeans ####
plot_kmeans_color(ax_1, img)


# plt.tight_layout()
# plt.subplots_adjust(left=0.5, bottom=None, right=None, top=0.9, wspace=None, hspace=None)
plt.savefig("{}.pdf".format(image_id), dpi=400)
plt.show()

