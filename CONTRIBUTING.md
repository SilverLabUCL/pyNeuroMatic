To develop pyNeuroMatic, you can use the `dev` extra and the `development` branch:

    git clone https://github.com/SilverLabUCL/pyneuromatic.git
    cd
    git checkout development
    pip install .[dev]

Please use pre-commit to run the pre-commit hooks.
You will need to install the hooks once:

    pre-commit install

They will then run before each commit.

We follow a pull request workflow.
