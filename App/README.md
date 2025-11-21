# 3D Cell Data Pipeline
The 3D Celldata Pipeline aims to automate cell data analysis by segmenting instances of, e.g., Cells or Nuclei.<br>
**David Exler**

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/DavidExler/Masterarbeit
cd Masterarbeit
```

### 2. Install Python  
Make sure you have Python (3.11+) installed.

- **Download Python:**  
  https://www.python.org/downloads/

Check your version:
```bash
python --version
```

### 3. Create & Activate a Virtual Environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

---

## Optional: Docker Setup

If you prefer using Docker, you can run the project inside a container.

- **Download Docker:**  
  https://www.docker.com/get-started/

## Data Preparation

There is a recommended data directory structure.  
Place a folder in the folder named `data/` for every new dataset:
This new folder is recommended to contain the subfolders 
[Input](./helpers/data/Demo/Input/), [Data](./helpers/data/Demo/Data/), [Models](./helpers/data/Demo/Models/), and [Output](./helpers/data/Demo/Output/) as follows:

```
project-root
│
├── helpers
│   ├── ... python_scripts.py ...
│   ├── data
│   │   ├── Demo dataset 1
│   │   │   ├── Data
│   │   │   ├── Input
│   │   │   │   ├── demo1.tif
│   │   │   │   ├── demo2.tif
│   │   │   │   └── ...
│   │   │   ├── Models 
│   │   │   ├── Output
│   │   └── Demo dataset 2
│   │   │   ├── ...
...
```
Place your Dataset inside the [Input](./helpers/data/Demo/Input/) folder. Data must be in .tif format and of shape (Z, X, Y, D). The files `./helpers/data/Demo/Input/demo.tif` can be arbitrarily named
## Usage
### 1. Run the script
```bash
python pipe.py 
```
## Optional: Docker Setup
### Build the Image
```bash
docker build -t masterarbeit .
```

### Run the Container
```bash
docker run -p 8000:8000 masterarbeit
```

### 2. Connect to the GUI
Visit http://localhost:8050/ 

### 3. Data Analysis
Start by entering your chosen dataset folder and click "Start Segmentation".
From here on, complete the pipeline procession step by step. Keep in mind, the data annotation does not need to be fully exhaustive.

### 4. Results
The results of the dataset are availabe in the [Output](./helpers/data/Demo/Input/) folder.