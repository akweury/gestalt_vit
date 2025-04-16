


### Train Command 

``` 
docker run -it  --gpus all  --shm-size=8g  -v /home/ml-jsha/gestalt_vit:/app  -v /home/ml-jsha/.netrc:/root/.netrc  -v /home/ml-jsha/storage/tiny-imagenet-200:/tiny-224 --rm --name adaptive_vit gvit 
```