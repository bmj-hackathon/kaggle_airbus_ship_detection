#%%
# data_path = Path("/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/ASSETS/Dogs vs Cats")
# image_path = data_path / '12499.jpg'
# image_path = data_path / '12500.jpg'
data_path = Path("/media/batman/f4023177-48c1-456b-bff2-cc769f3ac277/DATA/airbus-ship-detection")
assert data_path.exists()
img_zip_path = data_path / 'train_v2.zip'
assert img_zip_path.exists()
record_path = data_path / 'train_ship_segmentations_v2.csv'
assert record_path.exists()

img_zip = zipfile.ZipFile(img_zip_path)

logging.info("Image data: '{}' loaded from {} with {} files".format('img_zip', img_zip_path.name, len(img_zip.filelist) ))

#%%
df = pd.read_csv(record_path)

logging.info("{} with {} records".format(record_path.name, len(df)))
logging.info("{} unique file names found in df".format(df['ImageId'].unique().shape[0]))
# Flag if the record has a mask entry
df['HasShip'] = df['EncodedPixels'].notnull()
# Flag if the record is NOT unique
df['Duplicated'] = df['ImageId'].duplicated()
df['Unique'] = df['Duplicated']==False

logging.info("{} records with mask information (ship)".format(df['HasShip'].value_counts()[True]))
logging.info("{} images have at least one ship".format(sum(df['HasShip'] & df['Unique'])))

df_by_image = df.groupby('ImageId').agg({'HasShip': ['first', 'sum']})
df_by_image.columns = ['HasShip', 'TotalShips']
df_by_image.sort_values('TotalShips', ascending=False, inplace=True)
df = df.set_index('ImageId')
df_sample = df.head()
