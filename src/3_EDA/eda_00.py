#%% Image class

class Image():
    def __init__(self, image_id):
        """

        :param image_id:

        Attributes:
            image_id    The ID string
            img         The image as an ndarray
            records     DataFrame of records from the original CSV file
            encoding    A string representing the OpenCV encoding of the underlying img ndarray
        """
        self.image_id = image_id
        self.encoding = None
        self.records = None
        self.img = None

        logging.info("Image id: {}".format(self.image_id))

    def __str__(self):
        return "Image ID {} {} encoded, with {} ships".format(self.image_id, self.encoding, self.num_ships)

    def load(self, image_zip, df):
        self.img = imutils.load_rgb_from_zip(image_zip, image_id)
        self.encoding = 'RGB'
        logging.info("Loaded {}, size {} ".format(image_id, self.img.shape))

        # records = df.loc[[ image_id ]]
        self.records = df[df.index == self.image_id]
        assert isinstance(self.records, pd.DataFrame)

        # # Enforce dataframe
        # if type(records) == pd.core.series.Series:
        #     records = pd.DataFrame(records)
        #     records = records.T

        # assert len(records) == df_by_image.loc[image_id]['TotalShips']
        logging.info("{} records selected for {}".format(len(self.records), self.image_id))

    @property
    def num_ships(self):
        return len(self.records)

    def get_contours(self):
        # Iterate over each record
        contours = list()
        cnt=0
        for i, rec in records.iterrows():
            cnt+=1
            logging.debug("Processing record {} of {}".format(cnt, image_id))
            mask = imutils.convert_rle_mask(rec['EncodedPixels'])
            contour = imutils.get_contour(mask)
            contours.append(contour)
            # img = imutils.draw_ellipse_and_axis(img, contour, thickness=2)
            img = imutils.fit_draw_ellipse(img, contour, thickness=2)
        return img, contours




image_id = df_by_image.index[2] # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df)
print(image)



#%% Start a fig

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

