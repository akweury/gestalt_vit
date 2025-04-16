# gestalt_vit
ViT parameter reduction via Gestalt Principles

---
## Docker
``` 
docker build -t gestalt_vit:latest .

docker run -it --gpus all -v /home/ml-jsha/gestalt_vit:/app --rm gvit:latest
```

---
## Run Scripts

``` 
pip install -r requirements.txt

python -m src.train --device 10 

```

-d --gpus all --rm --shm-size=8g --name vit_tiny_train -v

/storage-01/datasets/tiny-imagenet-200-224/tiny-224