## Installation of the dependencies

1. Clone the repository:

```bash
git clone https://github.com/mattreturn1/AI_project.git
```

2. Download the library required

```bash
pip install -r requirements.txt
```

3. The repository contains the already filtered folders.  
   If you want to test the `folders_organizer.py`, you need to download `abide.zip` from the following link:

[https://auckland.figshare.com/articles/dataset/NeurIPS_2022_Datasets/21397377](https://auckland.figshare.com/articles/dataset/NeurIPS_2022_Datasets/21397377)

Then extract the contents of the `abide.zip` folder and rename the extracted folder to `abide`.    
Move the renamed folder to the root folder of the project.

## Executing the code

1. Calculate the metrics with `main.py`, which must be done before
   the execution of the other programs.

2. Then use Jupyter Notebook to execute
   `bayesian_network.ipynb` and `hidden_markov_model.ipynb`.
