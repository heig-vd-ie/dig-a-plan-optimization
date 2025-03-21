grid-synthesizer
=============

(Description to come...)

## Grid-synthesizer

### 1. Install pipx on wsl

```bash
sudo apt update
sudo apt install pipx
pipx ensurepath --force
```

### 2. Install python 3.12 on wsl (if not installed)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12
sudo apt install python3.12-venv
```

Reference: [Tutorial is the following link](https://www.linuxtuto.com/how-to-install-python-3-12-on-ubuntu-22-04/)

### 3. Install Poetry

```bash
pipx install poetry
```

### 4. Create virtual environment

```bash
python3.12 -m venv .venv
poetry env use .venv/bin/python3.12
```

### 5. Update .venv library

```bash
poetry update
```

> [!IMPORTANT]
> If psycopg-c installation raise the [error](https://stackoverflow.com/questions/77727508/problem-installing-psycopg2-for-python-venv-through-poetry): _psycopg-c (3.1.18) not supporting PEP 517 builds_

```bash
    sudo apt install libpq-dev gcc
```

### 6. [Download docker](https://www.docker.com/)

### 7. Create docker container

```bash
docker compose up
```

## Other command

### Check which python is installed

```bash
which python3.12
```

### Check python version

```bash
/usr/bin/python3 --version
```

or

```bash
python3.12 --version
```
### Check the poetry linked environment

```bash
poetry3.12 env info
```

### Check the Ubuntu

```bash
lsb_release -a
```

### Initialize pyproject.toml

```bash
poetry init
```