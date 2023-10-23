# Arquitetura de Software
Repositório para a disciplina de Arquitetura de Software - Grupo 6 (MLOps)

## Pré-requisitos

Antes de rodar o experimento, instale as depedências necessárias usando:

```
    pip install -r requirements.txt
```

Você também precisará instalar o [Docker](https://docs.docker.com/engine/install/)


## Experimento
Para rodar o experimento e observar os resultados você deve seguir os seguintes passos:

1. Iniciar o banco de dados
    ```
    sudo docker run -p 5432:5432 --name <container_name> -e POSTGRES_USER=<user> -e POSTGRES_PASSWORD=<password> -e POSTGRES_DB=<db> -d postgres
    ```
2. Iniciar o MLFlow server
    ```
    mlflow server --backend-store-uri postgresql://<user>:<password>@localhost/<db> --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001
    ```
3. Rodar um experimento:
    ```
    python3 random_forest.py
    ``` 
4. Rodar algumas das outras funcionalidades implementadas:
    
    Retreinar o modelo de produção
    ```
    python3 retrain_production_model.py --help
    ```
    
    Setar o melhor modelo como modelo de produção
    ```
    python3 set_best_model_to_production.py --help
    ```
    
    Rodar algum modelo específico de uma run:
    ```
    python3 load_from_specific_run.py --help
    ```

    Usar uma versão específica do modelo:
    ```
    python3 use_model_version.py
    ```
    

