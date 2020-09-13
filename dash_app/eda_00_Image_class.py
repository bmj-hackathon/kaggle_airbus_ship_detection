#%% Image class
import logging
import imutils
import numpy as np
import sklearn as sk
# import sklearn.cluster
import cv2
import pandas as pd
import base64

class SimpleImage:
    def __init__(self, image_id):
        pass


class Image():
    def __init__(self, image_id):
        """An object representing a full satellite image with ships and it's metadata.

        An image record is identified by the init parameter `image_id`.
        Images are stored in a zip file, and loaded on demand using the `.load(zip_path, df)`. method.
        The zip file path is needed, as well as a Dataframe object holding the metadata. The
        metadata for an image is the list of ship location records.

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
        if len(self.records) == 1:
            rec = self.records.head(1)
            if isinstance(rec['EncodedPixels'].values[0], str):
                return 1
            else:
                return 0
        else:
            return len(self.records)

    @property
    def shape(self):
        return self.img.shape

    @property
    def shape2D(self):
        return self.img.shape[0:2]

    def get_img_bgr(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

    def load(self, image_zip, df):
        """load an image into ndarray as RGB, and load ship records

        :param image_zip:
        :param df:
        :return:
        """

        self.img = imutils.load_rgb_from_zip(image_zip, self.image_id)
        # TODO: FOR TESTING ONLY!!!!
        # self.img = 200 * np.ones(shape=self.img.shape, dtype=np.uint8)
        # self.img[:300,:300,:] = 100 * np.ones(3, dtype=np.uint8)
        # logging.info("DEBUG ON: ".format())

        # print(self.img)
        # print(self.img.shape)

        self.encoding = 'RGB'
        logging.info("Loaded {}, size {} ".format(self.image_id, self.img.shape))

        # TODO: (Actually just a note: the .copy() will suppress the SettingWithCopyWarning!
        self.records = df.loc[df.index == self.image_id, :].copy()
        assert isinstance(self.records, pd.DataFrame)

        self.records['ship_id'] = self.records.apply(lambda row: hash(row['EncodedPixels']), axis=1)
        self.records.set_index('ship_id', inplace=True)
        self.records.drop(['HasShip', 'Duplicated', 'Unique'], axis=1, inplace=True)

        logging.info("{} records selected for {}".format(len(self.records), self.image_id))

    def moments(self):
        """ Just a docstring for now
            // spatial moments
    double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    // central moments
    double  mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    // central normalized moments
    double  nu20, nu11, nu02, nu30, nu21, nu12, nu03;
        :return:
        """

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

        # def get_x(row): return round(row['moments']['m10'] / row['moments']['m00'])
        def get_x(row): return row['moments']['m10'] / row['moments']['m00']
        # def get_y(row): return round(row['moments']['m01'] / row['moments']['m00'])
        def get_y(row): return row['moments']['m01'] / row['moments']['m00']
        self.records['x'] = self.records.apply(lambda row: get_x(row), axis=1)
        self.records['y'] = self.records.apply(lambda row: get_y(row), axis=1)

        # ( Same as m00!)
        self.records['area'] = self.records.apply(lambda row: cv2.contourArea(row['contour']), axis=1)
        self.records['rotated_rect'] = self.records.apply(lambda row: cv2.minAreaRect(row['contour']), axis=1)
        self.records['angle'] = self.records.apply(lambda row: row['rotated_rect'][2], axis=1)

    def ship_summary_table(self):
        if self.num_ships:
            df_summary = self.records.copy()
            df_summary.drop(['mask', 'contour', 'moments', 'rotated_rect', 'EncodedPixels'], axis=1, inplace=True)
            df_summary.reset_index(drop=True, inplace=True)
            df_summary.insert(0, 'ship', range(0, len(df_summary)))
            logging.info("Generating summary table".format())
            return df_summary.round(1)
        else:
            return None

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

    def draw_ellipses_img(self):
        logging.info("Fitting and drawing ellipses on a new ndarray canvas.".format())
        canvas = self.img
        for idx, rec in self.records.iterrows():
            # logging.debug("Processing record {} of {}".format(cnt, image_id))
            # contour = imutils.get_contour(rec['mask'])
            # img = imutils.draw_ellipse_and_axis(img, contour, thickness=2)
            # print(rec)
            # print(rec['contour'])
            canvas = imutils.fit_draw_ellipse(canvas, rec['contour'], thickness=2)
        return canvas

    def k_means(self, num_clusters=2):
        logging.info("Processing {} image of shape {}".format(self.encoding, self.img.shape))
        data = self.img / 255
        logging.info("Scaled values to 0-1 range".format())
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        logging.info("Reshape to pixel list {}".format(data.shape))

        kmeans = sk.cluster.MiniBatchKMeans(num_clusters)
        kmeans.fit(data)
        logging.info("Fit {} pixels into {} clusters".format(data.shape[0], num_clusters))
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        for c_name, c_count, c_position in zip(unique, counts, kmeans.cluster_centers_):
            logging.info("\tCluster {} at {} with {:0.1%} of the pixels".format(c_name, np.around(c_position, 3), c_count/data.shape[0])),

        if len(unique) == 2:
            dist = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
            logging.info("Distance between c1 and c2: {}".format(dist))
        return kmeans
        # all_new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

def fit_kmeans_pixels(img, kmeans):
    """

    :param img: An RGB image
    :param kmeans: The fit KMeans sci-kit object over this image
    :return: A new image, fit to the clusters of the image
    """
    original_pixels = img / 255
    # original_pixels = image.img.copy() / 255
    new_shape = original_pixels.shape[0] * original_pixels.shape[1], original_pixels.shape[2]
    original_pixels = original_pixels.reshape(new_shape)
    # logging.info("Reshape to pixel list {}".format(original_pixels.shape))
    # logging.info("Changed values to 0-1 range".format(img.shape))
    logging.info("Scaled image to [0-1] and reshaped to {}".format(new_shape))

    predicted_cluster = kmeans.predict(original_pixels)
    # TODO: Document this Numpy behaviour - indexing one array with an integer array of indices
    # Creates a new array, of length equal to indexing array
    # test_a = predicted_cluster[0:10]
    # test_asdf = kmeans.cluster_centers_
    # test_asdf[test_a]
    img_clustered_pixels = kmeans.cluster_centers_[predicted_cluster]
    logging.info("Assigned each pixel to a cluster (color vector).".format())

    img_clustered_pixels = img_clustered_pixels.reshape(img.shape)
    logging.info("Reshape pixels back to original shape".format())
    logging.info("Returning KMeans fit image canvas".format())
    # img_clustered_pixels
    return img_clustered_pixels


def convert_rgb_img_to_b64string_straight(img):
    # Encode the in-memory image to .jpg format
    retval, buffer = cv2.imencode('.jpg', img)

    # Convert to base64 raw bytes
    jpg_as_text = base64.b64encode(buffer)

    # Decode the bytes to utf
    jpg_as_text = jpg_as_text.decode(encoding="utf-8")

    logging.info("Image encoded to jpg base64 string".format())

    return jpg_as_text

def convert_rgb_img_to_b64string(img):
    # Convert image to BGR from RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Encode the in-memory image to .jpg format
    retval, buffer = cv2.imencode('.jpg', img)

    # Convert to base64 raw bytes
    jpg_as_text = base64.b64encode(buffer)

    # Decode the bytes to utf
    jpg_as_text = jpg_as_text.decode(encoding="utf-8")

    logging.info("Image encoded to jpg base64 string".format())

    return jpg_as_text


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


if 0: # DEV
    image_id = df_by_image.index[2] # Select an image with 15 ships
    image_id = df_by_image.index[-2]
    selfimage = Image(image_id)
    selfimage.records
    selfimage.load(img_zip, df)
    selfimage.load_ships()
    df_summary = selfimage.ship_summary_table()
    df_summary
    # for idx,  in selfimage.records.iterrows():
    #     print(i['contour'])

    # i
    # selfimage.records['contour']

    canvas2 = selfimage.draw_ellipses_img()

    # for i in
    #     print(i )

    # r = image.records