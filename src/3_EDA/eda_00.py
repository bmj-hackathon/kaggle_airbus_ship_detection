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
            ships       A list of Ship dictionary entries
                ship_id         - Hash of the RLE string
                EncodedPixels   - RLE string
                center          -
        """

        self.image_id = image_id
        self.encoding = None
        self.records = None
        self.img = None
        self.contours = None

        logging.info("Image id: {}".format(self.image_id))

    def __str__(self):
        return "Image ID {} {} encoded, with {} ships".format(self.image_id, self.encoding, self.num_ships)

    @property
    def num_ships(self):
        return len(self.records)

    @property
    def shape(self):
        return self.img.shape

    @property
    def shape2D(self):
        return self.img.shape[0:2]

    def load(self, image_zip, df):
        """load an image into ndarray as RGB, and load ship records

        :param image_zip:
        :param df:
        :return:
        """

        self.img = imutils.load_rgb_from_zip(image_zip, image_id)
        self.encoding = 'RGB'
        logging.info("Loaded {}, size {} ".format(image_id, self.img.shape))

        self.records = df[df.index == self.image_id]
        assert isinstance(self.records, pd.DataFrame)

        # TODO: check warning
        self.records['ship_id'] = self.records.apply(lambda row: hash(row['EncodedPixels']), axis=1)
        self.records.set_index('ship_id', inplace=True)
        self.records.drop(['HasShip', 'Duplicated', 'Unique'], axis=1, inplace=True)

        logging.info("{} records selected for {}".format(len(self.records), self.image_id))

    def load_ships(self):
        """Augment the basic df with mask, contour, data

        mask        - ndarray of 0 or 1
        contour     - opencv2 contour object
        moments     -

        :return:
        """
        assert isinstance(self.img, np.ndarray), "No image loaded"
        assert self.num_ships, "No ships in this image"

        # TODO: check warnings
        self.records['mask'] = self.records.apply(lambda row: self.convert_rle_to_mask(row['EncodedPixels'], self.shape2D), axis=1)
        self.records['contour'] = self.records.apply(lambda row: self.get_contour(row['mask']), axis=1)
        self.records['moments'] = self.records.apply(lambda row: cv2.moments(row['contour']), axis=1)
        self.records['area'] = self.records.apply(lambda row: cv2.moments(row['contour']), axis=1)

        def get_x(row): return round(row['moments']['m10'] / row['moments']['m00'])
        def get_y(row): return round(row['moments']['m01'] / row['moments']['m00'])

        self.records['x'] = self.records.apply(lambda row: get_x(row), axis=1)
        self.records['y'] = self.records.apply(lambda row: get_y(row), axis=1)

        # self.records['y'] = round(M['m01'] / M['m00'])
        # self.records['area'] = int(cv2.contourArea(c))
        # self.records['rotated_rect'] = cv2.fitEllipse(c)





        # Iterate over each record
        if 0:
            contours = list()
            cnt=0
            for i, rec in self.records.iterrows():
                cnt+=1
                # logging.debug("Processing record {} of {}".format(cnt, image_id))
                mask = self.convert_rle_to_mask(rec['EncodedPixels'], self.img.shape[0:2])
                contour = self.get_contour(mask)
                contours.append(contour)
                # img = imutils.draw_ellipse_and_axis(img, contour, thickness=2)
            # return img, contours

    def draw_ellipses_to_canvas(self):
        img = imutils.fit_draw_ellipse(self.img, contour, thickness=2)

    def convert_rle_to_mask(self, rle, shape):
        """convert RLE mask into 2d pixel array"""

        # Initialize a zero canvas (one-dimensional here)
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        # Split each run-length string
        s = rle.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            mask[start:start + length] = 1 # Assign this run to ones
        # Reshape to 2D
        img2 = mask.reshape(shape).T
        return img2

    def get_contour(self, mask):
        """Return a cv2 contour object from a binary 0/1 mask"""

        assert mask.ndim == 2
        assert mask.min() == 0
        assert mask.max() == 1
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(contours) == 1, "Too many contours in this mask!"
        contour = contours[0]
        # logging.debug("Returning {} fit contours over mask pixels".format(len(contours)))
        return contour


image_id = df_by_image.index[2] # Select an image with 15 ships
image = Image(image_id)
image.load(img_zip, df)
image.load_ships()
# print(image)
# image.get_contours()
r = image.records.transpose()
# hash(image.records['EncodedPixels'].values)



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

