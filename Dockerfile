FROM python:3.11-slim

WORKDIR /app

# 1. installer dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. copier tout le projet
COPY . .

# 3. générer le modèle dans le container
RUN python best_model.py


EXPOSE 3000

CMD ["bentoml", "serve", "service:svc", "--port", "3000"]