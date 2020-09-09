# polarbot

Dialog generation model including POLAR dataset language insight

# Installation

## Docker build

```
cd $PATH_TO_REPO/polarbot-nmt-rest/
docker build -t polarbot-responder-api:v0.0.2 .
```

## Docker run
```
docker run -it
  -e MODEL_PATH=/usr/src/app/models/$MODEL_FILENAME \
  -p $LOCAL_PORT:5000 \
  -v $MODEL_DIR:/usr/src/app/models \
  polarbot-responder-api:v0.0.2
```

# Example usage

```
curl -X POST "localhost:5005/api/translate" \
  -H "Content-type: application/json" \
  -d '{"text": "hi how are you"}'
```
