#%% Image class
import logging
import imutils
import numpy as np
import sklearn as sk
# import sklearn.cluster
import cv2
import pandas as pd

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

    def get_img_bgr(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

    def load(self, image_zip, df):
        """load an image into ndarray as RGB, and load ship records

        :param image_zip:
        :param df:
        :return:
        """

        self.img = imutils.load_rgb_from_zip(image_zip, self.image_id)
        self.encoding = 'RGB'
        logging.info("Loaded {}, size {} ".format(self.image_id, self.img.shape))

        self.records = df[df.index == self.image_id]
        assert isinstance(self.records, pd.DataFrame)

        # TODO: check warning
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
        # print(self.records['rotated_rect'])
        # print(type(self.records['rotated_rect'].iloc[0]))


        # self.records['area'] = int()
        # self.records['rotated_rect'] = cv2.fitEllipse(c)

        self.records.drop(['mask', 'contour', 'moments', 'rotated_rect', 'EncodedPixels'], axis=1, inplace=True)

    def ship_summary_table(self):
        return self.records.round(1)

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

    def k_means(self, num_clusters=2):
        logging.info("Processing {} image of shape {}".format(self.encoding, self.img.shape))
        data = self.img / 255
        logging.info("Scaled values to 0-1 range".format())
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        logging.info("Reshape to pixel list {}".format(data.shape))

        kmeans = sk.cluster.MiniBatchKMeans(2)
        kmeans.fit(data)
        logging.info("Fit {} pixels {} clusters".format(data.shape[0], num_clusters))
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        for c_name, c_count, c_position in zip(unique, counts, kmeans.cluster_centers_):
            logging.info("\tCluster {} at {} with {:0.1%} of the pixels".format(c_name, np.around(c_position, 3), c_count/data.shape[0])),

        if len(unique) == 2:
            dist = np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
            logging.debug("Distance between c1 and c2: {}".format(dist))
        return kmeans
        # all_new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

def convert_img_to_b64string(img):

