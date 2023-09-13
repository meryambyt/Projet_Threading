"""Alignment between sequence and structure.

This script generates an optimal alignment between a structure and 
a sequence using dual dynamic programming.

Usage:
======
    python main.py filename.pdb filename.fasta
"""

__authors__ = "Boulayat Meryam"
__date__ = "2023-09-13"

# Import librairies
from Bio import PDB
from Bio import SeqIO
import numpy as np
from functools import lru_cache
import multiprocessing
import argparse
import sys
import os


def three_to_one(three_letter_code):
    """
    Convert a three-letter amino acid code to one-letter code.

    Args:
        three_letter_code (str): Three-letter amino acid code.

    Returns:
        str: One-letter amino acid code or None if not found.
    """
    amino_acids = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    return amino_acids.get(three_letter_code.upper())


def extract_seq(fasta_file):
    """
    Extract the sequence from a FASTA file.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        str: The sequence as a string.

    Raises:
        ValueError: If the file is not in FASTA format.
        FileNotFoundError: If the file does not exist.
    """
    # Check if the file exists
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"The file {fasta_file} does not exist.")

    # Check the file extension to ensure it's a FASTA file
    _, file_extension = os.path.splitext(fasta_file)
    if file_extension.lower() != ".fasta":
        raise ValueError("The file is not in FASTA format.")

    # Open the FASTA file and read the sequence
    with open(fasta_file, "r") as f:
        # Use the SeqIO module from BioPython to read the sequence
        record = SeqIO.read(f, "fasta")
        # Extract the sequence as a string
        sequence = str(record.seq)

    return sequence


def extract_seq_from_pdb(pdb_file):
    """
    Extract the sequence from a PDB file.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        str: The sequence of the template.

    Raises:
        ValueError: If the file is not in PDB format.
        FileNotFoundError: If the file does not exist.
    """

    # Check if the file exists
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"The file {pdb_file} does not exist.")

    # Check the file extension to ensure it's a PDB file
    _, file_extension = os.path.splitext(pdb_file)
    if file_extension.lower() != ".pdb":
        raise ValueError("The file is not in PDB format.")

    # Open the PDB file and parse its contents
    with open(pdb_file, "r") as f:
        # Create a PDBParser instance and parse the PDB structure
        parser = PDB.PDBParser(PERMISSIVE=1)
        structure = parser.get_structure("structure", f)

    # Select chain "A"
    chain = structure[0]["A"]

    # Extract the sequence by converting three-letter residue
    # codes to one-letter codes
    sequence = "".join(
        [three_to_one(res.get_resname()) for res in chain.get_residues()]
    )

    return sequence


def extract_secondary_structure(pdb_file):
    """
    Extract secondary structure information from a PDB file.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        str: Secondary structure information mapped to the sequence.
    """
    # Create a PDBParser instance and parse the PDB structure
    parser = PDB.PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Create a DSSP object to calculate secondary structure
    model = structure[0]
    dssp = PDB.DSSP(model, pdb_file)

    # Get the sequence from the PDB file
    sequence = "".join([res.get_resname() for res in model.get_residues()])

    # Initialize secondary structure mapping
    ss_mapping = ""

    # Map secondary structure to the sequence
    for res_id in dssp.property_dict:
        ss_info = dssp.property_dict[res_id]
        ss = ss_info[2]  # Index 2 corresponds to secondary structure
        if ss == "H":
            ss_mapping += "H"  # Helix
        elif ss == "E":
            ss_mapping += "E"  # Beta sheet
        else:
            ss_mapping += " "  # Loop or other

    return ss_mapping


def extract_dope_score(dope_file):
    """
    Extract DOPE (Discrete Optimized Protein Energy) scores from a file.

    Args:
        dope_file (str): Path to the DOPE score file.

    Returns:
        dict: Dictionary of DOPE scores.
    """
    # Initialize an empty dictionary to store DOPE scores
    dope_dico = {}

    # Open the DOPE score file for reading
    with open(dope_file, "r") as f:
        # Iterate through each line in the file
        for line in f:
            # Split the line into words
            words = line.split()
            # Check if the line contains information about two CA
            # (carbon alpha) atoms
            if words[1] == "CA" and words[3] == "CA":
                # Convert three-letter residue codes to one-letter codes
                residue1 = three_to_one(words[0])
                residue2 = three_to_one(words[2])
                # Extract scores
                scores = [float(score) for score in words[4:]]

                # Create a key to identify the pair of residues
                key = f"{residue1} {residue2}"

                # Initialize a dictionary to store scores at
                # different distances
                score_each_distance = {}
                # Initial distance
                distance = 0.75

                # Store scores at different distances
                for score in scores:
                    score_each_distance[distance] = -score
                    distance += 0.5

                # Add the scores for this residue pair to the DOPE dictionary
                if key not in dope_dico:
                    dope_dico[key] = score_each_distance

    return dope_dico


def ca_coordinates(pdb_filename):
    """
    Extract C-alpha (CA) coordinates from a PDB file.

    Args:
        pdb_filename (str): Path to the PDB file.

    Returns:
        np.ndarray: Numpy array of CA coordinates.
    """
    # Initialize an empty list to store CA coordinates
    ca_coordinates = []

    # Create a PDBParser instance
    parser = PDB.PDBParser()
    # Parse the PDB structure from the file
    structure = parser.get_structure("template", pdb_filename)

    # Iterate through the PDB structure to extract CA coordinates
    for template in structure:
        for chain in template:
            for residue in chain:
                # Check if the object is a residue
                if isinstance(residue, PDB.Residue.Residue):
                    # Check if "CA" atom exists in the residue
                    if "CA" in residue:
                        # Append CA coordinates to the list
                        ca_coordinates.append(residue["CA"].coord)

    # Convert the list of CA coordinates to a numpy array
    ca_coordinates = np.array(ca_coordinates)

    return ca_coordinates


def dist_matrix_ca(ca_coords):
    """
    Calculate the distance matrix of C-alpha (CA) atoms.

    Args:
        ca_coords (np.ndarray): NumPy array of CA coordinates.

    Returns:
        np.ndarray: Distance matrix.
    """
    # Get the number of CA atoms
    n = ca_coords.shape[0]

    # Initialize a matrix filled with zeros
    dist_matrix = np.zeros((n, n))

    # Calculate distances between all pairs of CA atoms
    for i in range(n):
        for j in range(n):
            # Ensure we don't calculate the distance between the same atom
            if i != j:
                # Calculate euclidean distance between CA atoms
                dist_matrix[i, j] = np.linalg.norm(ca_coords[i] - ca_coords[j])

    # Return the distance matrix
    return dist_matrix


def find_nearest_key(dope_score_aa, target_value):
    """
    Key in DOPE score dictionary that corresponds to nearest value to target value.

    Args:
        dope_score_aa (dict): DOPE score dictionary for a specific amino acid pair.
        target_value (float): Target value for comparison.

    Returns:
        float: Nearest key in the DOPE score dictionary.
    """
    # Find the key in the dictionary that minimizes the absolute
    # difference with the target value
    nearest_key = min(dope_score_aa, key=lambda key: abs(key - target_value))
    return nearest_key


def init_low_level_matrix(position_sequence, position_template, 
                          sequence, template):
    """
    Initialize low-level matrix.

    Args:
        position_sequence (int): Position in the sequence.
        position_template (int): Position in the template.
        sequence (str): The input sequence.
        template (str): The sequence of the template.

    Returns:
        np.ndarray: Initialized low-level matrix.
    """
    # Increment sequence and template positions by 1
    position_sequence += 1
    position_template += 1

    # Get the lengths of the sequence and the template
    len_seq = len(sequence)
    len_template = len(template)

    # Initialize a matrix filled with zeros
    low_matrix = np.zeros((len_seq + 1, len_template + 1))

    # Fill the matrix with "inf" to not calculate unnecessary values
    for i in range(len_seq + 1):
        for j in range(len_template + 1):
            if (
                ((i >= position_sequence) and (j <= position_template))
                or ((i <= position_sequence) and (j >= position_template))
                or (i == position_sequence)
            ):
                low_matrix[i][j] = float("inf")

    # Fixed position equals 0
    low_matrix[position_sequence][position_template] = 0

    return low_matrix


def score_low_matrix(
    position_sequence, position_template, sequence, dope_score,
    dist_ca, gap_penalty
):
    """
    Calculate the score of the low-level dynamic programming matrix.

    Args:
        position_sequence (int): Position in the sequence.
        position_template (int): Position in the template.
        sequence (str): The input sequence from fasta file.
        dope_score (dict): Dictionary of DOPE scores.
        dist_ca (np.ndarray): Distance matrix of C-alpha atoms.

    Returns:
        float: Score of the specified cell in the matrix.
    """

    # Initialize the matrix with 0 in position for which the score will
    # be calculated and inf in positions for which the score will not
    #  be calculated
    low_matrix = init_low_level_matrix(
        position_sequence, position_template, sequence, dist_ca
    )

    # initialize the score
    last_score = None

    # Add 1 due to the addition of an extra row and column
    position_sequence += 1
    position_template += 1

    # Fill the first row
    for i in range(position_sequence):
        low_matrix[i][0] = gap_penalty * i

    # Fill the first column
    for j in range(position_template):
        low_matrix[0][j] = gap_penalty * j

    # Fill the first sub-matrix until fixed position
    for i in range(1, position_sequence):
        for j in range(1, position_template):
            # Fill only when value equals 0 (never calculates +inf cases)
            if low_matrix[i][j] == 0:
                # Extract pair of amino acid
                aa = f"{sequence[i - 1]} {sequence[position_sequence - 1]}"
                # Calculate distance between positons
                val_dist = find_nearest_key(
                    dope_score[aa], dist_ca[j - 1][position_template - 1]
                )
                # Extract corresponding dope score
                score = dope_score[aa][val_dist]

                # Apply Needleman and Wunsch algorithm
                match = low_matrix[i - 1][j - 1] + score
                north = low_matrix[i - 1][j] + gap_penalty
                west = low_matrix[i][j - 1] + gap_penalty
                low_matrix[i][j] = min(match, west, north)
                last_score = low_matrix[i][j]

    # Fill the second sub-matrix if it exists
    if ((position_sequence + 1) < low_matrix.shape[0]) and (
        (position_template + 1) < low_matrix.shape[1]
    ):
        # Extract the amino acid pair positioned diagonally from
        # the specified position.
        aa = f"{sequence[position_sequence]} {sequence[position_sequence]}"
        # Calculate distance between positons
        val_dist = find_nearest_key(
            dope_score[aa], dist_ca[position_template][position_template - 1]
        )
        # Extract the corresponding dope score
        score = dope_score[aa][val_dist]
        low_matrix[position_sequence + 1][position_template + 1] = (
            low_matrix[position_sequence - 1][position_template - 1] + score
        )

        # Calculate score for the rest of the matrix
        for i in range(position_sequence + 1, low_matrix.shape[0]):
            for j in range(position_template + 1, low_matrix.shape[1]):
                # Score positioned diagonally from sepcified position
                # was already calculated above, so continue
                if (i == position_sequence + 1) and (j == position_template + 1):
                    continue
                # Fill only when value equals 0 (never calculates +inf cases)
                if low_matrix[i][j] == 0:
                    aa = f"{sequence[i - 1]} {sequence[position_sequence - 1]}"
                    val_dist = find_nearest_key(
                        dope_score[aa], dist_ca[j - 1][position_template - 1]
                    )
                    score = dope_score[aa][val_dist]
                    match = low_matrix[i - 1][j - 1] + score
                    north = low_matrix[i - 1][j] + gap_penalty
                    west = low_matrix[i][j - 1] + gap_penalty
                    low_matrix[i][j] = min(match, west, north)
                    last_score = low_matrix[i][j]

    # If position_sequence = 0 and position_template = low_matrix.shape[1]
    # or position_sequence = low_matrix.shape[0] and position_template = 0
    # last_score will be equal to none since all values of the matrix equal
    # +inf
    if last_score is None:
        last_score = low_matrix[position_sequence - 1][position_template - 1] + 1

    # If the final score does not correspond to the bottom-right
    # corner of the matrix, it means that we have reached the edge
    # of the matrix in either the sequence or template direction.
    # In this case, we need to add gap penalties to the final score.
    if position_sequence == (low_matrix.shape[0] - 1):
        for j in range(position_template, low_matrix.shape[1]):
            last_score += gap_penalty
    if position_template == (low_matrix.shape[1] - 1):
        for i in range(position_sequence, low_matrix.shape[0]):
            last_score += gap_penalty

    return last_score


def calculate_matrix_element(args):
    """
    Calculate the matrix element in parallel.

    Args:
        args (tuple): Tuple of arguments for high matrix.

    Returns:
        float: Score of the specified cell in the matrix.
    """
    i, j, sequence, dope_score, dist_ca, gap_penalty = args
    return score_low_matrix(i, j, sequence, dope_score, dist_ca, gap_penalty)


def high_matrix(sequence, dist_ca, dope_score, gap_penalty):
    """
    Calculate high-level matrix containing scores.

    Args:
        sequence (str): The input sequence from fasta file.
        dist_ca (np.ndarray): Distance matrix of C-alpha atoms.
        dope_score (dict): Dictionary of DOPE scores.

    Returns:
        np.ndarray: High-level dynamic programming matrix.
    """
    # Calculate length of the input sequence and the template
    taille_seq = len(sequence)
    taille_template = len(dist_ca)

    # Initialize empty matrix to store scores
    matrix = np.zeros((taille_seq, taille_template))

    # Determine number of CPU cores for parallel processing
    num_cores = multiprocessing.cpu_count()

    # Create a multiprocessing pool for parallel computation
    pool = multiprocessing.Pool(processes=num_cores)

    # Generate a list of arguments for calculating matrix elements in parallel
    args_list = [
        (i, j, sequence, dope_score, dist_ca, gap_penalty)
        for i in range(taille_seq)
        for j in range(taille_template)
    ]

    # Use parallel processing to calculate matrix elements
    results = pool.map(calculate_matrix_element, args_list)

    # Close and join the multiprocessing pool
    pool.close()
    pool.join()

    # Fill the matrix with the scores
    for i in range(taille_seq):
        for j in range(taille_template):
            matrix[i][j] = results[i * taille_template + j]

    return matrix


def f_matrix(high_matrix, sequence, template_seq, gap_penalty):
    """
    Calculate the final alignment matrix and aligned sequences.

    Args:
        high_matrix (np.ndarray): High-level dynamic programming matrix.
        sequence (str): The input sequence from fasta file.
        template_seq : The sequence of the template.

    Returns:
        np.ndarray: Final alignment matrix.
        str: Aligned sequence.
        str: Aligned template.
    """
    # Initialize an empty matrix for the final alignment
    matrix = np.zeros((high_matrix.shape[0] + 1, high_matrix.shape[1] + 1))

    # Initialize the first row and column of the matrix with gap penalties
    for i in range(matrix.shape[0]):
        matrix[i][0] = gap_penalty * i

    for j in range(matrix.shape[1]):
        matrix[0][j] = gap_penalty * j

    # Calculate score with Needleman and Wunsch algorithm
    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            score = high_matrix[i - 1][j - 1]
            match = matrix[i - 1][j - 1] + score
            north = matrix[i - 1][j] + gap_penalty
            west = matrix[i][j - 1] + gap_penalty
            matrix[i][j] = min(match, west, north)

    seq = []
    template = []
    i, j = len(sequence), len(template_seq)

    # Trace back to find the aligned sequences
    while i > 0 and j > 0:
        diag = matrix[i - 1, j - 1] + high_matrix[i - 1, j - 1]
        haut = matrix[i - 1, j] + gap_penalty
        gauche = matrix[i, j - 1] + gap_penalty

        if i > 0 and matrix[i, j] == haut:
            seq.append(sequence[i - 1])
            template.append("-")
            i -= 1
        elif i > 0 and j > 0 and matrix[i, j] == diag:
            seq.append(sequence[i - 1])
            template.append(template_seq[j - 1])
            i -= 1
            j -= 1
        else:
            seq.append("-")
            template.append(template_seq[j - 1])
            j -= 1

    # Reverse and join the aligned sequences
    seq = "".join(seq[::-1])
    template = "".join(template[::-1])

    return matrix, seq, template


def banner():
    print("\n" + "-" * 80)
    print("\n")
    print("    /\\    | |(_)                                            | |  ")
    print("   /  \\   | | _   __ _  _ __    ___  _ __ ___    ___  _ __  | |_ ")
    print("  / /\\ \\  | || | / _` || '_ \\  / _ \\| '_ ` _ \\  / _ \\| '_ \\ | __|")
    print(" / ____ \\ | || || (_| || | | ||  __/| | | | | ||  __/| | | || |_ ")
    print("/_/    \\_\\|_||_| \\__, ||_| |_| \\___||_| |_| |_| \\___||_| |_| \\__|")
    print("                  __/ |                                           ")
    print("                 |___/                                            ")
    print("  _____  _                       _                      ")
    print(" / ____|| |                     | |                     ")
    print("| (___  | |_  _ __  _   _   ___ | |_  _   _  _ __   ___ ")
    print(" \\___ \\ | __|| '__|| | | | / __|| __|| | | || '__| / _ \\")
    print(" ____) || |_ | |   | |_| || (__ | |_ | |_| || |   |  __/")
    print("|_____/  \\__||_|    \\__,_| \\___| \\__| \\__,_||_|    \\___|")
    print("                                                        ")
    print("                                                        ")
    print("  _____                                              ")
    print(" / ____|                                             ")
    print("| (___    ___   __ _  _   _   ___  _ __    ___   ___ ")
    print(" \\___ \\  / _ \\ / _` || | | | / _ \\| '_ \\  / __| / _ \\")
    print(" ____) ||  __/| (_| || |_| ||  __/| | | || (__ |  __/")
    print("|_____/  \\___| \\__, | \\__,_| \\___||_| |_| \\___| \\___|")
    print("                  | |                                ")
    print("                  |_|                                ")
    print("\nOutil d'alignement de séquence et de structure")
    print("Version 1.0")
    print("Copyright © 2023 Meryam Boulayat")
    print(
        "\nUtilisation : python mon_script.py fichier_structure.pdb fichier_sequence.fasta"
    )
    print(
        "\nPour plus d'informations, visitez : https://github.com/meryambyt/Projet_Threading.git"
    )
    print("\n" + "-" * 80 + "\n")


def format_print_sequence(sequence, start):
    """
    Format sequence with a fixed width of 20 characters per line.

    Args:
        sequence (str): The input sequence to be formatted.
        start (int): The starting line number for formatting.

    Returns:
        str: The formatted sequence.
    """
    lines = [sequence[i : i + 20] for i in range(0, len(sequence), 20)]
    formatted_lines = []
    for i, line in enumerate(lines):
        line_number = start + i * 20
        formatted_line = f"{line_number:10d} {line}"
        formatted_lines.append(formatted_line)
    return "\n".join(formatted_lines)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description=
        "Align a FASTA sequence with a PDB structure using DOPE scores."
    )

    # Add arguments for FASTA and PDB files
    parser.add_argument("pdb_file", help="Path to the PDB file")
    parser.add_argument("fasta_file", help="Path to the FASTA file")

    # Add an argument to specify gap penalty
    parser.add_argument(
        "--gap_penalty", type=float, default=0, help="Gap penalty score (default: 0)"
    )

    # Add an argument to specify the output file name
    parser.add_argument(
        "--output",
        dest="output_file",
        default="output.log",
        help="Output file name (default: output.log)",
    )

    args = parser.parse_args()

    try:
        # Check if the FASTA and PDB files adhere to the format
        seq_list = extract_seq(args.fasta_file)
        template_seq = extract_seq_from_pdb(args.pdb_file)
    except Exception as e:
        print(f"Error reading files: {e}", file=sys.stderr)
        sys.exit(1)

    banner()
    dope_filename = "dope.par"
    template_seq = extract_seq_from_pdb(args.pdb_file)
    ca_coords = ca_coordinates(args.pdb_file)
    dist_ca = dist_matrix_ca(ca_coords)
    seq_list = extract_seq(args.fasta_file)
    ss = extract_secondary_structure(args.pdb_file)
    dope_score = extract_dope_score(dope_filename)
    print("Loading alignment matrix...\n")
    matrix = high_matrix(seq_list, dist_ca, dope_score, args.gap_penalty)
    print("Loading optimal alignment...\n")
    align_matrix, seq, template = f_matrix(
        matrix, seq_list, template_seq, args.gap_penalty
    )

    # Show only 20 amino acids per line
    total_lines = max(len(seq), len(template), len(ss)) // 20 + 1
    seq = seq.ljust(total_lines * 20)
    template = template.ljust(total_lines * 20)
    ss = ss.ljust(total_lines * 20)
    print("Optimal alignment result : \n")
    # Write results to the output.log file
    with open(args.output_file, "w") as output_file:
        # Print sequences and write them in a file
        for i in range(total_lines):
            start = i * 20
            end = (i + 1) * 20
            formatted_seq = format_print_sequence(seq[start:end], start)
            formatted_template = format_print_sequence(template[start:end], start)
            formatted_ss = format_print_sequence(ss[start:end], start)
            print(f"Sequence            : {formatted_seq}")
            print(f"Structure           : {formatted_template}")
            print(f"Secondary structure : {formatted_ss}")
            print("\n")

            output_file.write(f"Sequence            : {formatted_seq}\n")
            output_file.write(f"Structure           : {formatted_template}\n")
            output_file.write(f"Secondary structure : {formatted_ss}\n")
            output_file.write("\n")
