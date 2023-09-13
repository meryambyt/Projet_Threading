# Threading: alignment between a sequence and a structure using dual dynamic programming

This program is based on [THREADER by David Jones](https://www.sciencedirect.com/science/article/abs/pii/S0167730608604706?via%3Dihub). This tools does an alignment between a sequence and a pdb stucture by using dual dynamic programming. The program may take some time to execute the task even with multiprocessing depending on the sequence and structure length.


## Setup your environment

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

Clone this repository:

```bash
git clone https://github.com/meryambyt/Projet_Threading.git
```
Move to the new directory:

```bash
cd Projet_Threading
```

Create the `Projet_Threading` conda environment:
```
conda env create -f environnement.yml
```

Load the conda environment:
```
conda activate threading_boulayat
```

To deactivate a conda active environment, use

```
conda deactivate
```

## Run the program

To learn how to use the program, go to **/src/** folder and use the command:
```bash
python main.py -h
```

This will show the following message : 
```
usage: main.py [-h] [--gap_penalty GAP_PENALTY] [--output OUTPUT_FILE] pdb_file fasta_file

Align a FASTA sequence with a PDB structure using DOPE scores.

positional arguments:
  pdb_file              Path to the PDB file
  fasta_file            Path to the FASTA file

optional arguments:
  -h, --help            show this help message and exit
  --gap_penalty GAP_PENALTY
                        Gap penalty score (default: 0)
  --output OUTPUT_FILE  Output file name (default: output.log)
```

The input structure file has to be in **.pdb** format and the sequence in **.fasta** format. It is possible to choose a name for the output file by using the argument --output, but also changing the value of the gap penalty (default value equals 0). Here is an example :

```bash
python main.py filename.pdb filename.fasta --gap_penalty 2 --output alignement1.log
```
If you want to keep the default parameters, here is the command you have to run :
```bash
python main.py filename.pdb filename.fasta
```

This will print the alignment results on the terminal and create an output file **output.log**. Don't forget to specify the path in which you want the file to be located.

Test files are provided in the **/data/** folder for data reproducibility. The results of those data are discussed in the paper called **BOULAYAT_MERYAM_threading.pdf**.






