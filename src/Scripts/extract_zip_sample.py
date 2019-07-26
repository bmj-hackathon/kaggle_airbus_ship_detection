
#%%
import logging
from pathlib import Path
import zipfile
import random
data_path = Path("/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/DATA/airbus-ship-detection")
assert data_path.exists()
img_zip_path = data_path / 'train_v2.zip'
assert img_zip_path.exists()
record_path = data_path / 'train_ship_segmentations_v2.csv'
assert record_path.exists()

img_zip = zipfile.ZipFile(img_zip_path)

logging.info("Image data: '{}' loaded from {} with {} files".format('img_zip', img_zip_path.name, len(img_zip.filelist) ))

NUMBER_IMAGES = 20000
PATH_TARGET = Path('~/DATA/airbus-ship-detection-sample').expanduser()
assert PATH_TARGET.exists()
#%%
sample_list = random.sample(img_zip.namelist(),NUMBER_IMAGES)
# img_zip.extractall(PATH_TARGET,sample_list[0:10])
img_zip.extractall(PATH_TARGET,sample_list)
