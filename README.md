# kaggle_airbus_ship_detection

## Local
### Dependencies

Use poetry

Note a customized (forked) library; 

poetry add git+https://github.com/MarcusJones/imutils

poetry export --without-hashes -f requirements.txt > requirements.txt

### Containerization
docker build -t airbus-dash .

docker container run -it --entrypoint /bin/bash airbus-dash

docker run -v "/media/batman/3D6450257A2A5BEC1/00 DATA/DATA/airbus-ship-detection":/data -t -p 80:80 airbus-dash

docker save my_image > my_image.tar

## Remote
### Install docker
```
sudo usermod -a -G docker $USER
newgrp docker
```

### Download the data set
```
mkdir .kaggle
vim ~/.kaggle/kaggle.json
mkdir data
cd data
~/.local/bin/kaggle competitions download -c airbus-ship-detection
```

### Get the image loaded
```
scp -i KEY.pem -r "airbus-dash.tar" ubuntu@DNS:~

scp -i KEY.pem -r "airbus-dash.tar" ubuntu@DNS:~

sudo docker load < airbus-dash.tar

docker run -v "/home/ubuntu/data":/data -t -p 80:80 airbus-dash
```
