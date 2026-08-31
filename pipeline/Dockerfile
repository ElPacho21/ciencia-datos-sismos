FROM astrocrpublic.azurecr.io/runtime:3.3-4

# La lógica de ingesta vive en include/sismos y se importa como `sismos`.
# Sin esta línea el paquete existe en la imagen pero no está en el path.
ENV PYTHONPATH="/usr/local/airflow/include:${PYTHONPATH}"
