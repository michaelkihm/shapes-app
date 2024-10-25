# shapes-app

## Gettings started

```bash
cp secrets.env.template secrets.env
docker-compose up
```

## Training

- Create Weights and Biases [account](https://wandb.ai/)
- Copy WandB API key in paste in [secrets](secrets.env) env file as WANDB_API_KEY.
- Restart docker file with docker-compose up
- Run train.py

```bash
python train.py
```