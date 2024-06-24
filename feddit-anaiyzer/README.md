# Feddit Analyzer

Web API that identifies if comments on a given subfeddit or category are positive or negative.

## Model Choice

The model used to provide the sentiment analysis functionality is [tweeter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

The decision to use this model was based on the following reasons:
    - The model seems to perform adequately in sentiment analysis tasks.
    - The model is pre-trained on a large corpus of tweets, which may have a similar content and
        format to Reddit comments which Feedit aims to emulate.
    - There is a serverless API available for prototyping that allows to call the model, which makes
        it easy to use and convenient.

In a real world scenario, the model may be fine-tuned on an internal dataset of labeled Feddit
comments and deployed properly using a cloud service like AWS SageMaker Endpoints. Adding all the
model observability and monitoring tools to ensure the model is working as expected.

### Hugging Face API

You will need to provide a Hugging Face API key to use the model through the environment variable ``HUGGINGFACE_API_KEY``.

During development that can be done by creating a ``.env`` file at the root of the repositor, which will be ignored by Git or setting the environment variable in CodeSpaces.

## Dependencies

* [Docker](https://docs.docker.com): Used to develop insider Docker development container and run several services in Docker Compose.

## Project setup

First step is to clone the repository.

```shell
git clone https://github.com/dainelli98/feddit-analyzer.git
```

After having cloned the repo, there are 2 ways to manage the development environment:

* Use a devcontainer (preferred):
  * Open the project in a development container with VSCode locally.
  * Open the project in a development container with VSCode hosted on GitHub Codespaces.
* Create a virtual environment and install Python packages:

```shell
make venv
```

> [!WARNING]
> Users of Mac with ARM processors will need to default to develop within GitHub Codespaces or another alternative environments. This is due to ``chasingcars/feddit`` not being compatible with those devices.
>
> ```shell
> $ docker pull chasingcars/feddit:latest
> latest: Pulling from chasingcars/feddit
> no matching manifest for linux/arm64/v8 in the manifest list entries
> ```

To configure the coverage threshold, go to the ``.coveragerc`` file.

## Poetry

To include a new package to the project it should be added to ``pyproject.toml`` under the correct group:

* Packages needed to run the applications should be under the ``tool.poetry.dependencies`` section.
* Packages used as development tools such as ``pytest``, ``ruff`` or ``black`` belong to the ``tool.poetry.group.dev.dependencies`` section.

To add a package you can use ``poetry add``. You can indicate the group to add the dependency to with the option ``--group=GROUP``.

To remove a package use ``poetry remove``.

### poetry.lock

The ``poetry.lock`` file contains a snapshot of the resolved dependencies from ``pyproject.toml``.

To manually force the update of `poetry.lock` file, run ``poetry lock``. The ``--no-update`` flag can be used to avoid updating those dependencies which do not need to.

### Jupyter

You can start a local notebook or jupyter lab server with:

```shell
make jupyter
```

## Passwords

Some applications connect to any third-party service (for example a SQL server) requiring
a user name and a password.

This kind of information shall be never be hardcoded in the code or saved in any configuration
file that may be uploaded to the repository.

A simple way to handle critical data is saving them as environment variables.

Simply create a `.env` file at the root of the repository. Then and save user names and passwords
like:

```shell
YOUR_USERNAME=your_username
YOUR_PASSWORD=your_password
```

You can then read `.env` files for Python code with the `dotenv` package.

`.env` files are excluded from the Git repository in the `.gitignore` file.

## Testing

To verify correct installation, execute the Feddit Analyzer tests by running the following command in your Python environment:

```shell
make tests-basic
```

If you want to run all tests, that will require using Docker Compose:

```shell
docker compose -f feddit-api/test-docker-compose.yml up --build --abort-on-container-exit
```

## Generate documentation

Run the following command to generate project documentation:

```shell
make docs
```

## Contributors

* Martín Martínez, Daniel ([danitiana98@gmail.com](mailto:danitiana98@gmail.com))
