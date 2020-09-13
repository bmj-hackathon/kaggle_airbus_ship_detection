# kaggle_airbus_ship_detection

## Dependencies

Use poetry

Note a customized (forked) library; 

poetry add git+https://github.com/MarcusJones/imutils

poetry export --without-hashes -f requirements.txt > requirements.txt

## Containerization

docker build -t airbus-dash .

docker container run -it --entrypoint /bin/bash airbus-dash

```
docker run -v "/media/batman/3D6450257A2A5BEC1/00 DATA/DATA/airbus-ship-detection":/data -t airbus-dash
```