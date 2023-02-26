# pyDrop toolkit to process data from ddPCR

## Getting Started with Development [INCLOMPLETE]
1. Clone the repo from github
    ```sh
    git clone https://github.com/connordavel/pyDrop.git
    cd pyDrop
    ```
2. Create a development conda build 
    ```sh
    conda env create --name pydrop_dev -f doc/envs/pydrop_dev.yml
    conda activate pydrop_dev
    ```
3. Make a developent install of the pyDrop module
    ```sh
    cd pyDrop
    pip install -e .
    ```
4. Test run the build to make sure everything is working
    ```sh
    cd ..
    python tests/<future test file here>.py
    ```
    The above should (when finished) give you more info on your build and how to troubleshoot common issues if present (If they exist). 