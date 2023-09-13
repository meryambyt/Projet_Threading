from Bio import PDB
from Bio import SeqIO
import numpy as np

# Pour compatibilité avec code à une lettre de la sequence fasta et facilier la procedure
def three_to_one(three_letter_code):
    amino_acids = {
        'ALA': 'A',
        'ARG': 'R',
        'ASN': 'N',
        'ASP': 'D',
        'CYS': 'C',
        'GLN': 'Q',
        'GLU': 'E',
        'GLY': 'G',
        'HIS': 'H',
        'ILE': 'I',
        'LEU': 'L',
        'LYS': 'K',
        'MET': 'M',
        'PHE': 'F',
        'PRO': 'P',
        'SER': 'S',
        'THR': 'T',
        'TRP': 'W',
        'TYR': 'Y',
        'VAL': 'V'
    }
    return amino_acids.get(three_letter_code.upper())


# Parseur de fichiers fasta
def extract_seq(fasta_file):
    # Encore une fois verifier les conditions d'un fichier fasta

    # Initialisation d'un dictionnaire qui stocke le residu et 
    # numero de la position du residu dans la séquence
    seq_list = []

    with open(fasta_file, "r") as f:
        record = SeqIO.read(f, "fasta")
        sequence = str(record.seq)

    for i, char in enumerate(sequence):
        seq_list.append((i, char))

    return seq_list

# Extrait la séquence a partir du fichier pdb
def extract_seq_from_pdb(pdb_file):

    template_seq = []
    # Ouvrir le fichier PDB
    with open(pdb_file, "r") as f:
        parser = PDB.PDBParser(PERMISSIVE=1)
        structure = parser.get_structure("structure", f)

    # Obtenir la séquence de la chaîne de référence
    chain = structure[0]["A"]
    sequence = "".join([three_to_one(res.get_resname()) for res in chain.get_residues()])
    
    for i, char in enumerate(sequence):
        template_seq.append((i, char))

    # Retourner la séquence et les positions
    return template_seq



def extract_dope_score(dope_file):
    # Initialisation du dictionnaire contenant score DOPE
    dope_dico =  {}
    with open(dope_file, "r") as f:
        for line in f:
            words = line.split()
            # Prends uniquement en compte les carbones alpha des residus
            if words[1] == "CA" and words[3] == "CA":
                residue1 = three_to_one(words[0])
                residue2 = three_to_one(words[2])
                scores = [float(score) for score in words[4:]]

                key = f"{residue1} {residue2}"
                
                # Initialisation du dictionnaire contenant score pour chaque distance
                score_each_distance = {}
                distance = 0.75

                for score in scores:
                    score_each_distance[distance] = score
                    distance += 0.5
                
                if key not in dope_dico:
                    dope_dico[key] = score_each_distance
                    
    return dope_dico

# Permet à partir du fichier pdb d'extraire les coordonnées des carbones alpha        
def ca_coordinates(pdb_filename):
    # Mettre des conditions qui permettent de vérifier que le fichier donné en entrée est bien un fichier pdb
    # Liste permettant de stocker les coordonnées des carbones alpha
    ca_coordinates = []

    parser = PDB.PDBParser()
    structure = parser.get_structure("template", pdb_filename)

    for template in structure:
        for chain in template:
            for residue in chain:
                # Permet de verifier que l'element est bien un residu
                if isinstance(residue, PDB.Residue.Residue):
                    if "CA" in residue:
                        ca_coordinates.append(residue["CA"].coord)
    # Conversion de la liste en numpy array
    ca_coordinates = np.array(ca_coordinates)
    return ca_coordinates


# Matrice de distance des carbones alpha des residus du template
def dist_matrix_ca(ca_coords):

    # Nombre de carbones alpha
    n = ca_coords.shape[0]
    # Matrice de distance initialisée à zero
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i,j] = np.linalg.norm(ca_coords[i] - ca_coords[j])
    return dist_matrix


# Trouve la clé du dictionnaire qui correspond à la valeur la plus proche de distance
def find_nearest_key(dope_score_aa, target_value):

    nearest_key = min(dope_score_aa, key=lambda key: abs(key - target_value))
    return nearest_key


# Fonction qui permet d'initiliser les valeurs à 0 pour les cases qui seront calculées
# Les cases non calculées sont fixés à +infini. 
# Apres faire une fonction qui recupere la sequence comme ca juste seq_template au lieu de dist_ca
def init_low_level_matrix(position_sequence, position_template, seq_list, dist_ca):
    len_seq = len(seq_list)
    len_template = len(dist_ca)
    # Initialisation de la matrice de bas niveau
    low_matrix = np.zeros((len_seq, len_template))

    for i in range(len_seq):
        for j in range(len_template):
            if ((i >= position_sequence) and (j <= position_template)) or ((i <= position_sequence) and 
                                                                           (j >= position_template)) or (i == position_sequence):
                low_matrix[i][j] = float("inf")

    low_matrix[position_sequence][position_template] = 0
    print(low_matrix)
    return low_matrix


# Prend en entrée la matrice de bas niveau initialisée et la matrice de distance
def score_low_level(low_matrix, dist_ca, position_sequence, position_template, seq_list, dope_score):

    first_aa = f"{seq_list[0][1]} {seq_list[position_sequence][1]}"
    val_dist = find_nearest_key(dope_score[first_aa], dist_ca[0][position_sequence])
    low_matrix[0][0] = dope_score[first_aa][val_dist]

    # Remplir la premiere colonne
    for i in range(1,position_sequence):
         aa = f"{seq_list[i][1]} {seq_list[position_sequence][1]}"
         val_dist = find_nearest_key(dope_score[aa], dist_ca[0][position_sequence])
         low_matrix[i][0] = low_matrix[i-1][0] + dope_score[aa][val_dist]

    # Remplir la premiere ligne
    for j in range(1, position_template):
        aa = f"{seq_list[0][1]} {seq_list[position_sequence][1]}"
        val_dist = find_nearest_key(dope_score[aa], dist_ca[j][position_sequence])
        low_matrix[0][j] = low_matrix[0][j-1] + dope_score[aa][val_dist]

    # Replissage de toutes les cases restantes
    for i in range(1, len(seq_list)):
        for j in range(1, len(dist_ca)):
            if low_matrix[i][j] == 0:
                aa = f"{seq_list[i][1]} {seq_list[position_sequence][1]}"
                diag = low_matrix[i-1, j-1] + dope_score[aa][val_dist]
                nord = low_matrix[i-1, j] + dope_score[aa][val_dist]
                ouest = low_matrix[i, j-1] + dope_score[aa][val_dist]
                val_dist = find_nearest_key(dope_score[aa], dist_ca[j][position_sequence])
                low_matrix[i][j] =  min(diag, nord, ouest)

    return low_matrix[len(seq_list)-1][len(dist_ca)-1]


def high_matrix(seq_list, dist_ca, dope_score):

    taille_seq = len(seq_list)
    taille_template = len(dist_ca)
    # Initialisation de la matrice de haut niveau
    matrix = np.zeros((len(seq_list), len(dist_ca)))

    # Premiere case
    low_matrix_init = init_low_level_matrix(0, 0, seq_list, dist_ca)
    matrix[0,0] = score_low_level(low_matrix_init, dist_ca, 0, 0, seq_list, dope_score)
    

    # # Remplir la premiere colonne
    for i in range(1,taille_seq):
         low_matrix_init = init_low_level_matrix(i, 0, seq_list, dist_ca)
         score = score_low_level(low_matrix_init, dist_ca, i, 0, seq_list, dope_score)
         matrix[i][0] = matrix[i-1][0] + score

    # # Remplir la premiere ligne
    for j in range(1,taille_template):
         low_matrix_init = init_low_level_matrix(0, j, seq_list, dist_ca)
         score = score_low_level(low_matrix_init, dist_ca, 0, j, seq_list, dope_score)
         matrix[0][j] = matrix[0][j-1] + score

    for i in range(1,taille_seq):
        for j in range(1,taille_template):
            low_matrix_init = init_low_level_matrix(i, j, seq_list, dist_ca)
            matrix[i][j] = score_low_level(low_matrix_init, dist_ca, i, j, seq_list, dope_score)
            diag = matrix[i-1, j-1] + score
            nord = matrix[i-1, j] + score
            ouest = matrix[i, j-1] + score
            matrix[i][j] =  min(diag, nord, ouest)
    print("\n MATRICE FINAL\n")
    print(matrix)
    return matrix

"""
def alignement(high_matrix, seq_list, template_seq):
    # Initialisez les pointeurs au coin inférieur droit de la matrice
    i = len(seq_list) - 1
    j = len(template_seq) - 1

    # Initialisez les séquences d'alignement
    alignment_seq_list = []
    alignment_template_seq_list = []

    # Parcourez la matrice de bas en haut et de droite à gauche pour retracer l'alignement optimal
    while i >= 0 and j >= 0:
        diag_value = high_matrix[i - 1][j - 1] if i - 1 >= 0 and j - 1 >= 0 else float('inf')
        nord_value = high_matrix[i - 1][j] if i - 1 >= 0 else float('inf')
        ouest_value = high_matrix[i][j - 1] if j - 1 >= 0 else float('inf')

        # Trouvez la valeur minimale parmi les trois
        min_value = min(diag_value, nord_value, ouest_value)

        if min_value == diag_value:
            # Déplacez-vous en diagonale (match ou mismatch)
            alignment_seq_list.append(seq_list[i][1])  # Utilisez la lettre de l'acide aminé de seq_list
            alignment_template_seq_list.append(template_seq[j][1])  # Utilisez la lettre de l'acide aminé de template_seq
            i -= 1
            j -= 1
        elif min_value == nord_value:
            # Déplacez-vous vers le nord (gap dans seq_list)
            alignment_seq_list.append(seq_list[i][1])  # Utilisez la lettre de l'acide aminé de seq_list
            alignment_template_seq_list.append('-')  # Ajoutez un gap pour template_seq
            i -= 1
        else:
            # Déplacez-vous vers l'ouest (gap dans template_seq)
            alignment_seq_list.append('-')  # Ajoutez un gap pour seq_list
            alignment_template_seq_list.append(template_seq[j][1])  # Utilisez la lettre de l'acide aminé de template_seq
            j -= 1

    # Inversez les séquences d'alignement car nous les avons construites à partir de la fin
    alignment_seq_list = alignment_seq_list[::-1]
    alignment_template_seq_list = alignment_template_seq_list[::-1]

    # Convertissez les listes en chaînes
    alignment_seq = ''.join(alignment_seq_list)
    alignment_template_seq = ''.join(alignment_template_seq_list)

    return alignment_seq, alignment_template_seq
"""





if __name__ == "__main__":
    pdb_filename = "test.pdb"
    fasta_filename = "test.fasta"
    dope_filename = "dope.par"
    template_seq = extract_seq_from_pdb(pdb_filename)
    print(template_seq)
    ca_coords = ca_coordinates(pdb_filename)
    dist_ca = dist_matrix_ca(ca_coords)
    seq_list = extract_seq(fasta_filename)
    dope_score = extract_dope_score(dope_filename)
    # Test
    matrix = high_matrix(seq_list, dist_ca, dope_score)
    print(matrix)
    seq, structure = alignement(matrix, seq_list, template_seq)
    print(seq, "\n")
    print(structure, "\n")



    
"""
    def high_matrix(seq_list, dist_ca, dope_score):

    taille_seq = len(seq_list)
    taille_template = len(dist_ca)
    # Initialisation de la matrice de haut niveau
    matrix = np.zeros((len(seq_list), len(dist_ca)))
    gap_penalty = 0

    # Remplir la premiere ligne
    for i in range(taille_seq):
        matrix[i][0] = gap_penalty * i

    # Remplir la premiere colonne
    for j in range(0, taille_template):
        matrix[0][j] = gap_penalty * j

    # Remplir les autres cases
    for i in range(1, taille_seq):
        for j in range(1, taille_template):
            match_score = score_low_matrix(i, j, seq_list, dope_score, dist_ca)
            match = matrix[i - 1][j - 1] + match_score
            north = matrix[i - 1][j] + gap_penalty
            west = matrix[i][j - 1] + gap_penalty
            matrix[i][j] = min(match, west, north)
            print(i, j, "match :", match, " nord : ", north, " west ", west)
            print("\n MATRICE FINAL\n")
            print(matrix)
     
    return matrix
"""

[[ -1.77  -2.43   0.68  -2.98  -4.06  -2.94  -3.28  -3.55  -1.04   0.34    1.13   0.09 -33.46   -inf]
 [ -9.47 -15.67 -14.27 -15.7  -16.68 -15.82 -15.99 -16.03 -15.41 -14.54  -13.75 -13.73 -35.19   -inf]
 [ -8.66 -15.38 -13.25 -16.03 -17.86 -16.99 -15.91 -17.55 -15.92 -14.1  -13.31 -12.51 -24.05   -inf]
 [ -8.95 -18.92 -17.35 -19.6  -20.26 -20.87 -19.57 -20.73 -20.24 -18.94  -17.92 -16.9  -17.55   -inf]
 [  -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf    -inf   -inf   -inf   0.  ]]

[[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.65]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21  1.25]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21  1.25]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21  2.64]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21  2.64]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21
  3.06]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21
  3.06]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21
  3.06]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21
  3.06]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21
  3.06]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 0.21
  3.53]
 [0.   0.   0.   0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.21 0.21 3.09
  3.66]
 [0.   0.   0.   0.39 0.8  0.8  0.8  1.32 1.37 1.37 1.37 1.4  2.24 3.09
  5.84]]




def alignement(high_matrix, seq_list, template_seq):
    # Initialisez les pointeurs au coin inférieur droit de la matrice
    i = high_matrix.shape[0] - 1
    j = high_matrix.shape[1] - 1

    # Create variables to store alignment
    seq = ""
    template = ""


    # We'll use i and j to keep track of where we are in the matrix, just like above
    while i > 0 and j > 0: # end touching the top or the left edge
        score_current = high_matrix[i][j]
        score_diagonal = high_matrix[i-1][j-1]
        score_up = high_matrix[i][j-1]
        score_left = high_matrix[i-1][j]
        match_score = score_low_matrix(i-1, j-1, seq_list, dope_score, dist_ca)
        gap_penalty = -2
        # Check to figure out which cell the current score was calculated from,
        # then update i and j to correspond to that cell.

        if score_current == score_diagonal + match_score:
            seq += seq_list[i-1][1]
            template += template_seq[j-1][1]
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty :
            seq += seq_list[i-1][1]
            template += '-'
            i -= 1
        elif score_current == score_left + gap_penalty :
            seq += '-'
            template += template_seq[j-1][1]
            j -= 1

    
    # Since we traversed the score matrix from the bottom right, our two sequences will be reversed.
    # These two lines reverse the order of the characters in each sequence.
    seq = seq[::-1]
    template = template[::-1]

    return seq, template


[[ 0.000e+00 -2.000e+00 -4.000e+00 -6.000e+00 -8.000e+00 -1.000e+01 -1.200e+01 -1.400e+01 -1.600e+01 -1.800e+01 -2.000e+01 -2.200e+01  -2.400e+01 -2.600e+01       -inf]
 [-2.000e+00  2.000e-02 -1.980e+00 -3.980e+00 -5.980e+00 -7.980e+00  -9.950e+00 -1.179e+01 -1.367e+01 -1.567e+01 -1.751e+01 -1.920e+01  -2.120e+01 -2.320e+01       -inf]
 [-4.000e+00 -1.980e+00  4.000e-02 -1.960e+00 -3.960e+00 -5.930e+00  -7.850e+00 -9.710e+00 -1.151e+01 -1.351e+01 -1.524e+01 -1.646e+01  -1.815e+01 -2.015e+01       -inf]
 [-6.000e+00 -3.980e+00 -1.960e+00  6.000e-02 -1.940e+00 -3.940e+00  -5.930e+00 -7.820e+00 -9.590e+00 -1.159e+01 -1.327e+01 -1.435e+01  -1.557e+01 -1.757e+01       -inf]
 [-8.000e+00 -5.980e+00 -3.960e+00 -1.940e+00  8.000e-02 -1.890e+00  -3.890e+00 -5.890e+00 -7.640e+00 -9.640e+00 -1.123e+01 -1.296e+01  -1.404e+01 -1.604e+01       -inf]
 [-1.000e+01 -7.980e+00 -5.960e+00 -3.940e+00 -1.920e+00  1.600e-01  -1.810e+00 -3.730e+00 -5.590e+00 -7.590e+00 -9.010e+00 -1.000e+01  -1.173e+01 -1.373e+01       -inf]
 [-1.200e+01 -9.980e+00 -7.960e+00 -5.940e+00 -3.920e+00 -1.840e+00   1.900e-01 -1.790e+00 -3.660e+00 -5.660e+00 -7.410e+00 -8.170e+00  -9.160e+00 -1.116e+01       -inf]
 [-1.400e+01 -1.198e+01 -9.960e+00 -7.940e+00 -5.920e+00 -3.840e+00  -1.810e+00  2.300e-01 -1.610e+00 -3.610e+00 -5.300e+00 -6.310e+00  -7.070e+00 -9.070e+00       -inf]
 [-1.600e+01 -1.398e+01 -1.196e+01 -9.940e+00 -7.920e+00 -5.840e+00  -3.810e+00 -1.720e+00  4.300e-01 -1.570e+00 -3.300e+00 -4.510e+00  -5.520e+00 -7.520e+00       -inf]
 [-1.800e+01 -1.598e+01 -1.396e+01 -1.194e+01 -9.920e+00 -7.840e+00  -5.810e+00 -3.720e+00 -1.570e+00  1.800e-01 -1.390e+00 -2.500e+00  -3.710e+00 -5.710e+00       -inf]
 [-2.000e+01 -1.798e+01 -1.596e+01 -1.394e+01 -1.192e+01 -9.840e+00  -7.790e+00 -5.600e+00 -3.390e+00 -1.820e+00  6.700e-01 -5.900e-01  -1.700e+00 -3.700e+00       -inf]
 [-2.200e+01 -1.998e+01 -1.796e+01 -1.594e+01 -1.392e+01 -1.184e+01  -9.790e+00 -7.600e+00 -5.390e+00 -3.610e+00 -1.330e+00  1.770e+00   5.100e-01 -1.490e+00       -inf]
 [-2.400e+01 -2.198e+01 -1.996e+01 -1.794e+01 -1.592e+01 -1.384e+01  -1.176e+01 -9.600e+00 -7.300e+00 -5.490e+00 -2.980e+00 -1.000e-01   3.000e+00  1.000e+00       -inf]
 [      -inf       -inf       -inf       -inf       -inf       -inf        -inf       -inf       -inf       -inf       -inf       -inf        -inf       -inf  0.000e+00]
 [      -inf       -inf       -inf       -inf       -inf       -inf        -inf       -inf       -inf       -inf       -inf       -inf        -inf       -inf       -inf]]




[[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  -inf]
 [0.   0.02 0.02 0.02 0.02 0.02 0.05 0.21 0.33 0.33 0.49 0.8  0.8  0.8  -inf]
 [0.   0.02 0.04 0.04 0.04 0.07 0.15 0.29 0.49 0.49 0.76 1.54 1.85 1.85  -inf]
 [0.   0.02 0.04 0.06 0.06 0.07 0.15 0.29 0.49 0.49 0.76 1.65 2.43 2.43  -inf]
 [0.   0.02 0.04 0.06 0.08 0.11 0.15 0.29 0.49 0.49 0.85 1.65 2.43 2.43  -inf]
 [0.   0.02 0.04 0.06 0.08 0.16 0.19 0.31 0.59 0.59 1.12 2.08 2.88 2.88  -inf]
 [0.   0.02 0.04 0.06 0.08 0.16 0.19 0.31 0.59 0.59 1.12 2.08 2.92 2.92  -inf]
 [0.   0.02 0.04 0.06 0.08 0.16 0.19 0.31 0.59 0.59 1.12 2.22 3.18 3.18  -inf]
 [0.   0.02 0.04 0.06 0.08 0.16 0.19 0.31 0.59 0.59 1.12 2.22 3.18 3.18  -inf]
 [0.   0.02 0.04 0.06 0.08 0.16 0.19 0.31 0.59 0.59 1.12 2.22 3.18 3.18  -inf]
 [0.   0.02 0.04 0.06 0.08 0.16 0.21 0.4  0.64 0.64 1.12 2.22 3.18 3.18  -inf]
 [0.   0.02 0.04 0.06 0.08 0.16 0.21 0.4  0.64 0.64 1.12 2.22 3.32 3.32  -inf]
 [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf  0.  ]
 [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf  -inf]
 [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf  -inf]]


[[  0.    -1.    -2.     -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf   -inf]
 [ -1.     0.65  -0.35   -inf   -inf   -inf   -inf   -inf   -inf   -inf    -inf   -inf   -inf   -inf   -inf]
 [  -inf   -inf   -inf   0.     -inf   -inf   -inf   -inf   -inf   -inf    -inf   -inf   -inf   -inf   -inf]
 [  -inf   -inf   -inf   -inf -10.35 -11.35   0.     0.     0.     0.    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]]

[[  0.    -1.    -2.     -inf   -inf   -inf   -inf   -inf   -inf   -inf    -inf   -inf   -inf   -inf   -inf]
 [ -1.     0.65  -0.35   -inf   -inf   -inf   -inf   -inf   -inf   -inf    -inf   -inf   -inf   -inf   -inf]
 [  -inf   -inf   -inf   0.     -inf   -inf   -inf   -inf   -inf   -inf    -inf   -inf   -inf   -inf   -inf]
 [  -inf   -inf   -inf   -inf -10.35 -11.35   0.     0.     0.     0.   0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.   0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]
 [  -inf   -inf   -inf   -inf   0.     0.     0.     0.     0.     0.
    0.     0.     0.     0.     0.  ]]


[[  0.    -1.    -2.    -3.    -4.    -5.    -6.    -7.    -8.    -9.  -10.   -11.   -12.   -13.   -14.  ]
 [ -1.    -2.    -3.    -4.    -5.    -6.    -7.    -8.    -9.   -10.  -11.   -12.   -13.   -14.      nan]
 [ -2.    -3.    -4.    -5.    -6.    -7.    -8.    -9.   -10.   -11.  -12.   -13.   -14.   -15.   -16.  ]
 [ -3.    -4.    -5.    -6.    -7.    -8.    -9.   -10.   -11.   -12.  -13.   -14.   -15.   -16.   -17.  ]
 [ -4.    -5.    -6.    -7.    -8.    -9.   -10.   -11.   -12.   -13.  -14.   -15.   -16.   -17.   -18.  ]
 [ -5.    -6.    -7.    -8.    -9.   -10.   -11.   -12.   -13.   -14.  -15.   -16.   -17.   -18.   -19.  ]
 [ -6.    -7.    -8.    -9.   -10.   -11.   -12.   -13.   -14.   -15.  -16.   -17.   -18.   -19.   -20.  ]
 [ -7.    -8.    -9.   -10.   -11.   -12.   -13.   -14.   -15.   -16.  -17.   -18.   -19.   -20.   -21.  ]
 [ -8.    -9.   -10.   -11.   -12.   -13.   -14.   -15.   -16.   -17.  -18.   -19.   -20.   -21.   -22.  ]
 [ -9.   -10.   -11.   -12.   -13.   -14.   -15.   -16.   -17.   -18.  -19.   -20.   -21.   -22.   -23.  ]
 [-10.   -11.   -12.   -13.   -14.   -15.   -16.   -17.   -18.   -19.  -20.   -21.   -22.   -23.   -24.  ]
 [-11.   -12.   -13.   -14.   -15.   -16.   -17.   -18.   -19.   -20.  -21.   -22.   -23.   -24.   -23.92]
 [-12.   -13.   -14.   -15.   -16.   -17.   -18.   -19.   -20.   -21.  -22.   -23.   -24.   -25.   -23.49]
 [-13.   -14.   -15.   -16.   -17.   -18.   -19.   -20.   -21.   -22.  -23.   -24.   -25.   -24.61 -23.  ]
 [-14.      nan -16.   -17.   -18.   -19.   -20.   -21.   -22.   -23.  -24.   -25.   -26.   -25.61 -24.  ]]



[[ -9.26 -11.1  -11.75 -16.13 -18.95 -19.87 -21.49 -23.73 -25.45 -26.35  -28.4  -31.2  -34.   -26.  ]
 [-11.16 -11.52  -9.14 -13.3  -16.7  -17.53 -19.04 -20.98 -22.74 -24.03  -25.46 -28.41 -31.46 -24.35]
 [ -5.76  -5.89  -2.56  -4.2   -7.34  -8.39 -10.   -12.37 -13.8  -14.8  -15.79 -18.13 -21.01 -21.75]
 [-15.23 -15.13 -12.01 -11.84 -12.92 -14.6  -16.25 -17.95 -19.87 -20.87  -22.61 -24.87 -27.37 -20.17]
 [-11.6  -11.58  -7.6   -7.24  -7.3   -7.04  -7.96  -9.83 -11.08 -11.82  -12.93 -15.81 -18.77 -17.04]
 [-19.6  -19.61 -15.98 -14.88 -13.82 -13.35 -13.16 -14.49 -15.57 -16.42  -17.42 -19.65 -22.71 -15.58]
 [-16.5  -16.54 -12.87 -11.57 -10.44  -9.39  -7.81  -9.25  -8.57  -9.38   -9.96 -12.3  -14.74 -12.15]
 [-17.96 -18.01 -14.25 -13.19 -11.65 -10.6   -9.26  -8.79  -7.66  -7.97   -8.6  -10.04 -12.74 -10.81]
 [-25.54 -25.57 -22.14 -20.61 -18.96 -17.77 -15.4  -14.62 -12.03 -11.29  -12.13 -13.79 -16.31  -8.98]
 [-27.14 -27.16 -23.54 -21.73 -20.29 -19.32 -17.3  -16.37 -14.62 -11.82  -11.32 -12.24 -14.82  -7.26]
 [-29.54 -29.54 -26.18 -24.65 -22.84 -21.76 -18.74 -17.7  -15.01 -12.71  -10.39 -10.15 -12.8   -4.92]
 [-31.76 -31.76 -28.57 -26.6  -24.8  -23.68 -20.91 -19.8  -17.27 -15.18  -13.25  -9.77 -10.74  -2.49]
 [-34.   -34.   -31.37 -29.04 -27.21 -26.12 -22.95 -21.84 -19.67 -17.58  -15.6  -12.42  -0.61   0.  ]
 [-26.   -27.   -24.61 -22.32 -20.34 -19.31 -16.49 -15.53 -12.93 -10.83   -8.82  -5.82  -3.86  -1.94]]