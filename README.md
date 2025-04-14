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