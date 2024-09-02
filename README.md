## Search Back

Backend del proyecto final para la materia Recuperación de Información - 2024A EPN

El proyecto consiste en una aplicación web que permita a los usuarios ingresar una consulta y obtener una lista de imágenes relacionadas a la consulta.

El backend fue desarrollado con [FastAPI](https://fastapi.tiangolo.com/)

## Cómo usar

Para instalar el proyecto se debe clonar el repositorio, instalar las dependencias y ejecutar el servidor de desarrollo.

```bash

git clone https://github.com/Cheveniko/search-back.git
cd search-back
python3 -m venv venv
source venv/bin/activate # Este comando es diferente en Windows
pip3 install -r requirements.txt
fastapi dev app/main.py

```

## Autores

- [@Cheveniko](https://github.com/Cheveniko)
- [@Pinkylml](https://github.com/Pinkylml)

## Frontend

El frontend del proyecto donde se encuentra la interfaz de usuario y se consume el backend se encuentra en el siguiente repositorio: [Search Front](https://github.com/Cheveniko/search-front)
