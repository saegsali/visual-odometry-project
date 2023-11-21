# visual-odometry-project

## Installation
1. **Clone the repository**:  
    First, clone the repository either using SSH or HTTPS:
    Using SSH:

    ```
    git clone git@github.com:saegsali/visual-odometry-project.git
    ```
    Using HTTPS:
    ```
    git clone https://github.com/saegsali/visual-odometry-project.git
    ```
2. **Create a conda environment with the necessary dependencies**:  
    ```
    cd visual-odometry-project
    conda env create -f environment.yml
    ```
3. **Install the Visual Odometry Package:**
Install in editable mode using `-e` flag.
    ```
    pip install -e . 
    ```

### Dataset
You can use the setup script to download the dataset:
```
chmod +x setup.sh
./setup.sh
```
This will download all the datasets and extract them to the `data` folder.
