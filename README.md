
# Language-Guided Image Editing (LGIE)

## Running Locally

0. Install the requirements in ``requirements``

1. Change the ssh crudentials (USER_NAME and PASSWORD) in ``run_remote_single_GIER_jwt.sh``

2. Change the PROJECT_PATH and PYTHON_PATH in ``run_remote_single_GIER_jwt.sh`` and ``LGIE/test_single_GIER.sh``

3. run the following command 

    ```bash
    python manage.py makemigrations
    ```
    
    ```bash
    python manage.py migrate
    ```
    
    ```bash
    python manage.py runserver 
    ```


