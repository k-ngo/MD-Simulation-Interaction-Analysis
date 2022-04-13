import subprocess as sp
import os
import pandas as pd
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.ticker import AutoMinorLocator
from plip.structure.preparation import PDBComplex
from plip.basic import config
from warnings import simplefilter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
matplotlib.use('Agg')
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#################################################################################################################################
# Areas Formed By Opposing Oxygen Atoms in the Selectivity Filter (SVGFG)  or Select C_alpha Atoms in the Pore (res Y, F, or V) #
#################################################################################################################################
# Calculate areas formed by oxygen atoms of the same residue in opposing subunits (A = pi * r_segAC * r_segBD)
# in the selectivity filter (residues SVGFG) or select C_alpha atoms in the pore (res Y, F, or V) over the course of simulation.
# Requirement:
# VMD needs to be set so that it can be opened by invoking "vmd" in the terminal.
# Usage:
# 1. Run this script (Python 3)
#   python3 filter_and_pore_changes.py  -p [.psf file (default=.psf in current dir)]
#                                       -d [.dcd file (default=.dcd in current dir)]
#                                       -t [simulation time (default=1000)]
#                                       -o [location where graph will be saved (default='filter_and_pore_area.png')]
#   Optional arguments:
#                                       -n [plot title (default='Selectivity Filter and Pore Areas')]
#                                       -x [x label (default='Time (ns)')]
#                                       -s [label size (default=20)]

# Configurations
PDB_column_width_format = [(0, 4), (4, 11), (11, 16), (16, 20), (20, 22), (22, 26), (26, 38), (38, 46), (46, 54), (54, 60), (60, 66), (66, 90)]
parser = argparse.ArgumentParser(description='Time Series of Intermolecular Interactions')
parser.add_argument('-p', '--psf',
                    default=glob.glob('*.psf')[0],
                    dest='psf', action='store',
                    help='.psf file containing protein structural information')
parser.add_argument('-d', '--dcd',
                    default=glob.glob('*.dcd')[0],
                    dest='dcd', action='store',
                    help='.dcd file containing simulation trajectory (any trajectory format will also work)')
parser.add_argument('-s1',
                    dest='seg1', action='store',
                    help='First segment/chain/subunit to consider for analysis (invoked by segname in VMD)')
parser.add_argument('-s2',
                    dest='seg2', action='store',
                    help='Second segment/chain/subunit to consider for analysis (invoked by segname in VMD)')
parser.add_argument('-s1n',
                    dest='seg1_name', action='store',
                    help='Name of first segment/chain/subunit to consider for analysis (customized by user)')
parser.add_argument('-s2n',
                    dest='seg2_name', action='store',
                    help='Name of second segment/chain/subunit to consider for analysis (customized by user)')
parser.add_argument('-s1x',
                    default='',
                    dest='seg1_exclude', action='store',
                    help='Exclude this VMD selection from the first segment/chain/subunit')
parser.add_argument('-s2x',
                    default='',
                    dest='seg2_exclude', action='store',
                    help='Exclude this VMD selection from the second segment/chain/subunit')
parser.add_argument('-t', '--time',
                    default=777,
                    dest='time_total', action='store', type=float,
                    help='total simulation time of the full provided trajectory')
parser.add_argument('-e', '--end',
                    default=-1,
                    dest='end_frame', action='store', type=int,
                    help='analyze to which frame of the simulation')
parser.add_argument('-s', '--step',
                    default=1,
                    dest='step', action='store', type=int,
                    help='step used when loading trajectory (i.e., 1 to read every single frame, 2 to read every two frames...)')
parser.add_argument('--intramolecular',
                    dest='intramolecular', action='store_true',
                    help='account for intramolecular interactions in analysis')
parser.add_argument('--split',
                    default=1,
                    dest='split', action='store', type=int,
                    help='split each plot into # of smaller plots covering different time periods, useful for long simulations')
parser.add_argument('-x', '--xlabel',
                    default='Time (ns)',
                    dest='x_label', action='store',
                    help='label on x-axis')
parser.add_argument('--skipcommand',
                    dest='skip_command', action='store_true',
                    help='skip running VMD commands to generate input data, only set if the script has already been ran at least once')
parser.add_argument('--labelsize',
                    default=20,
                    dest='size', action='store', type=float,
                    help='label font size (default = 20)')
arg = parser.parse_args()

# Conversion of 3-letter amino acid code to 1-letter code
aa_names = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
            'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
            'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
            'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

# Thresholds for detection (global variables)
config.BS_DIST = 7.5  # Determines maximum distance to include binding site residues
config.AROMATIC_PLANARITY = 5.0  # Determines allowed deviation from planarity in aromatic rings
config.MIN_DIST = 0.5  # Minimum distance for all distance thresholds
config.HYDROPH_DIST_MAX = 4.0  # Distance cutoff for detection of hydrophobic contacts
config.HBOND_DIST_MAX = 3.5  # Max. distance between hydrogen bond donor and acceptor (Hubbard & Haider, 2001)
config.HBOND_DON_ANGLE_MIN = 100  # Min. angle at the hydrogen bond donor (Hubbard & Haider, 2001) + 10
config.PISTACK_DIST_MAX = 5.5  # Max. distance for parallel or offset pistacking (McGaughey, 1998)
config.PISTACK_ANG_DEV = 30  # Max. Deviation from parallel or perpendicular orientation (in degrees)
config.PISTACK_OFFSET_MAX = 2.0  # Maximum offset of the two rings (corresponds to the radius of benzene + 0.5 A)
config.PICATION_DIST_MAX = 6.0  # Max. distance between charged atom and aromatic ring center (Gallivan and Dougherty, 1999)
config.SALTBRIDGE_DIST_MAX = 5.5  # Max. distance between centers of charge for salt bridges (Barlow and Thornton, 1983) + 1.5


def get_interactions(resID, site, type):
    """Return residue numbers of residues involved in specified type of interactions"""
    if type == 'hbonds':
        return list(set([get_res_properties(i) for i in site.hbonds_ldon if abs(resID - i.resnr) != 4] + [get_res_properties(i) for i in site.hbonds_pdon if abs(resID - i.resnr) != 4]))
    elif type == 'hydrophobic_contacts':
        return list(set([get_res_properties(i) for i in site.hydrophobic_contacts]))
    elif type == 'water_bridges':
        return list(set([get_res_properties(i) for i in site.water_bridges]))
    elif type == 'salt_bridges':
        return list(set([get_res_properties(i) for i in site.saltbridge_lneg] + [get_res_properties(i) for i in site.saltbridge_pneg]))
    elif type == 'pi_stacking':
        return list(set([get_res_properties(i) for i in site.pistacking]))
    elif type == 'pi_cation':
        return list(set([get_res_properties(i) for i in site.pication_laro] + [get_res_properties(i) for i in site.pication_paro]))
    elif type == 'all':
        return get_interactions(site, 'hbonds') + get_interactions(site, 'hydrophobic_contacts') + get_interactions(site, 'water_bridges') + get_interactions(site, 'salt_bridges') + get_interactions(site, 'pi_stacking') + get_interactions(site, 'pi_cation')
    else:
        print('Invalid interaction type specified for', type)
        print('Select one of the followings:', ['hbonds', 'hydrophobic_contacts', 'water_bridges', 'salt_bridges', 'pi_stacking', 'pi_cation'])
        exit(1)


def get_res_properties(interaction):
    """From interaction, return interacting residue in format of <resname><resid>:<chain>"""
    return interaction.restype.replace(interaction.restype, aa_names.get(interaction.restype)) + str(interaction.resnr) + ':' + interaction.reschain


def closest(input_list, k):
    """Find closest number to k in list"""
    input_list = np.asarray(input_list)
    index = (np.abs(input_list - k)).argmin()
    return index


# Automatically determine input file name if given wildcard as input - will take first result that appears as input
if arg.psf.split('.')[0] == '*':
    arg.psf = glob.glob('*.' + arg.psf.split('.')[-1])[0]
if arg.dcd.split('.')[0] == '*':
    arg.dcd = glob.glob('*.' + arg.dcd.split('.')[-1])[0]

# If there are atoms to be excluded, generate VMD commands to do so
seg1_exclude = ''
seg2_exclude = ''
if arg.seg1_exclude:
    seg1_exclude = ' and not (' + str(arg.seg1_exclude) + '))'
if arg.seg2_exclude:
    seg2_exclude = ' and not (' + str(arg.seg2_exclude) + '))'

# Print input information
print('PSF         :', arg.psf)
print('DCD         :', arg.dcd)
print('1st Segment :', arg.seg1_name, '-', arg.seg1)
if arg.seg1_exclude:
    print('              excluding', str(arg.seg1_exclude))
print('2nd Segment :', arg.seg2_name, '-', arg.seg2)
if arg.seg2_exclude:
    print('              excluding', str(arg.seg2_exclude))

name = str(arg.seg1_name) + '_' + str(arg.seg2_name)
file_name = '.'.join(arg.dcd.split('.')[:-1])
interactions_pdb = os.path.join('temp_pdb', str(arg.seg1_name) + '_' + str(arg.seg2_name) + '.pdb')

# Create folders to store data and output
os.makedirs('temp_pdb', exist_ok=True)
os.makedirs('saved_results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Check required files
for script in ['prot_center.tcl']:
    if not os.path.exists(script):
        print('ERROR: Required script', script, 'not found in current directory.')
        exit(1)

# Load simulation trajectory and extract data
vmd_cmd_file = file_name + '_vmd_cmd.tcl'

if not arg.skip_command:
    with open(vmd_cmd_file, 'w+') as f:
        # Load trajectory files
        f.write('mol new ' + arg.psf + ' type psf first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n')
        f.write('mol addfile ' + arg.dcd + ' type ' + arg.dcd.split('.')[-1] + ' first 0 last ' + str(arg.end_frame) + ' step ' + str(arg.step) + ' filebonds 1 autobonds 1 waitfor all\n')

        # Center protein
        f.write('source prot_center.tcl\n')

        # Set interacting segment names
        f.write('set seg1 ' + str(arg.seg1) + '\n')
        f.write('set seg2 ' + str(arg.seg2) + '\n')

        # Obtain atoms from segment of interest
        f.write('set sel [atomselect top "(segname $seg1' + seg1_exclude + ') or (segname $seg2' + seg2_exclude + ')"]\n')
        f.write('animate write pdb ' + interactions_pdb + ' skip 1 sel $sel\n')

        # Initialize files to store data
        f.write('set outputname_seg1 "' + os.path.join('temp_pdb', name + '_interacting_residues_seg1.dat') + '"\n')
        f.write('set outputname_seg2 "' + os.path.join('temp_pdb', name + '_noninteracting_residues_seg2.dat') + '"\n')

        f.write('set nf [molinfo top get numframes]\n')
        f.write('set out1 [open ${outputname_seg1} w]\n')
        f.write('set out2 [open ${outputname_seg2} w]\n')

        f.write('for {set f 0} {$f < $nf} {incr f} {\n')
        # Loop through each frame to look at residues from seg1 that interact with those from seg2
        f.write('set interacting_resid_seg1 [atomselect top "(segname $seg1' + seg1_exclude + ') and name CA and same residue as within ' + str(config.BS_DIST) + ' of (segname $seg2' + seg2_exclude + ')" frame $f]\n')
        # as well as residues from seg2 that do not interact with those from seg1
        f.write('set noninteracting_resid_seg2 [atomselect top "(segname $seg2' + seg2_exclude + ') and name CA and not same residue as within ' + str(config.BS_DIST) + ' of (segname $seg1' + seg1_exclude + ')" frame $f]\n')

        f.write('puts $out1 "[$interacting_resid_seg1 get resid]"\n')
        f.write('puts $out2 "[$noninteracting_resid_seg2 get resid]" }\n')

        f.write('close $out1\n')
        f.write('close $out2\n')

        f.write('exit')

    sp.call(['/bin/bash', '-i', '-c', 'vmd -dispdev text -e ' + vmd_cmd_file], stdin=sp.PIPE)

# Obtain number of atoms and atom list
num_atoms = 0
with open(interactions_pdb) as f:
    next(f)  # Skip header
    for line in f:
        if 'END' in line:
            break
        num_atoms += 1

# Read IDs of residues from seg1 that can possibly interact with seg2
with open(os.path.join('temp_pdb', name + '_interacting_residues_seg1.dat')) as f:
    interacting_resid_seg1 = [list(map(int, i.split())) for i in f.read().splitlines()]

# Read IDs of residues from seg2 that cannot possibly interact with seg1
with open(os.path.join('temp_pdb', name + '_noninteracting_residues_seg2.dat')) as f:
    noninteracting_resid_seg2 = [list(map(int, i.split())) for i in f.read().splitlines()]

# Set up interaction maps
# ['H-bonds', 'Hydrophobic', 'Water bridges', 'Salt bridges', 'π/cation-π']
hbonds_map = pd.DataFrame(columns=['No such contacts encountered'])
hydrophobic_map = pd.DataFrame(columns=['No such contacts encountered'])
salt_bridges_map = pd.DataFrame(columns=['No such contacts encountered'])
pication_pi_map = pd.DataFrame(columns=['No such contacts encountered'])

# Analyze interactions throughout simulation trajectory
frame_count = 0
print('\nAnalyzing molecular interactions...')
# Look for previous results first, if not saved then proceed with calculations
saved_results = [os.path.join('saved_results', file_name + '_' + figure + '_' + name + '.csv') for figure in ['hbonds', 'hydrophobic', 'salt_bridges', 'pication_pi']]
if not all(os.path.exists(result) for result in saved_results):
    for frame in pd.read_fwf(interactions_pdb, chunksize=num_atoms + 1, header=None, colspecs=PDB_column_width_format, skiprows=1):
        # Delete last line (containing just 'END'), reset index to 0, and name columns
        frame.drop(frame.index[[-1]], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        frame.columns = ['Atom', 'Index', 'Type', 'Residue', 'Chain', 'ResNum', 'X', 'Y', 'Z', 'Occupancy', 'TempFactor', 'Segment']
        # Change water numbering to all integers
        frame.loc[frame['Residue'] == 'TIP', 'ResNum'] = frame['Index']
        # Convert float to int for select columns
        frame['ResNum'] = frame['ResNum'].astype(int)
        # frame.loc[frame['Residue'] != 'TIP', 'SeqNum'] += 401
        frame['Index'] = frame['Index'].astype(int)
        # Convert residues from seg1 that interact with those from seg2 to HETATM to serve as "ligands" for analysis
        for res in interacting_resid_seg1[frame_count]:
            frame.loc[(frame['ResNum'] == res) & (frame['Segment'] == arg.seg1), 'Atom'] = 'HETATM'
        # Remove residues from seg2 that do not interact with residues from seg1
        for res in noninteracting_resid_seg2[frame_count]:
            frame.drop(frame.loc[(frame['ResNum'] == res) & (frame['Segment'] == arg.seg2)].index, inplace=True)
        # If simulation is not set to analyze intramolecular interactions, remove all residues from seg1 that do not interact with seg2
        if not arg.intramolecular:
            frame.drop(frame.loc[(frame['Atom'] == 'ATOM') & (frame['Segment'] == arg.seg1)].index, inplace=True)
        # Convert "HSD" to "HIS"
        frame.loc[frame['Residue'] == 'HSD', 'Residue'] = 'HIS'
        # Fix chain name
        frame['Chain'] = frame['Segment'].str[-1]
        frame.loc[frame['Chain'] == '3', 'Chain'] = 'W'
        # Change segment column to element symbol
        frame['Segment'] = frame['Type'].str[0]
        # Convert every column to string for easier processing
        frame = frame.astype(str)
        # Record interactions
        pdb = PDBComplex()
        frame = '\n'.join([row['Atom'].ljust(6) + row['Index'].rjust(5) + row['Type'].center(5) + row['Residue'].rjust(4) +
                           row['Chain'].rjust(2) + row['ResNum'].rjust(4) + row['X'].rjust(12) + row['Y'].rjust(8) +
                           row['Z'].rjust(8) + row['Occupancy'].rjust(6) + row['TempFactor'].rjust(6) +
                           row['Segment'].rjust(12) for index, row in frame.iterrows()])
        pdb.load_pdb(frame, as_string=True)
        pdb.analyze()
        for key_res, site in pdb.interaction_sets.items():
            # Convert key residue from 3 letter code to 1 letter code
            key_res = key_res.replace(key_res.split(':')[0], aa_names.get(key_res.split(':')[0])).split(':')
            # Get resID. Convert key residue from format "<resname>:<resid>:<chain>" to "<resname><resid>:<chain>"
            key_resID = int(key_res[-1])
            key_res = key_res[0] + key_res[-1] + ':' + key_res[1]
            # Evaluate all interactions involving this key residue
            if len(get_interactions(key_resID, site, 'hbonds')) > 0:
                # Record interaction pair to dataframe
                for partner_res in get_interactions(key_resID, site, 'hbonds'):
                    interaction_pair = key_res + '-' + partner_res
                    hbonds_map.loc[frame_count, interaction_pair] = 1
                if 'No such contacts encountered' in hbonds_map.columns.values:
                    hbonds_map.drop(['No such contacts encountered'], axis=1, inplace=True)

            if len(get_interactions(key_resID, site, 'hydrophobic_contacts')) > 0:
                for partner_res in get_interactions(key_resID, site, 'hydrophobic_contacts'):
                    interaction_pair = key_res + '-' + partner_res
                    hydrophobic_map.loc[frame_count, interaction_pair] = 1
                if 'No such contacts encountered' in hydrophobic_map.columns.values:
                    hydrophobic_map.drop(['No such contacts encountered'], axis=1, inplace=True)

            if len(get_interactions(key_resID, site, 'salt_bridges')) > 0:
                for partner_res in get_interactions(key_resID, site, 'salt_bridges'):
                    interaction_pair = key_res + '-' + partner_res
                    salt_bridges_map.loc[frame_count, interaction_pair] = 1
                if 'No such contacts encountered' in salt_bridges_map.columns.values:
                    salt_bridges_map.drop(['No such contacts encountered'], axis=1, inplace=True)

            if len(get_interactions(key_resID, site, 'pi_stacking') + get_interactions(key_resID, site, 'pi_cation')) > 0:
                for partner_res in get_interactions(key_resID, site, 'pi_stacking') + get_interactions(key_resID, site, 'pi_cation'):
                    interaction_pair = key_res + '-' + partner_res
                    pication_pi_map.loc[frame_count, interaction_pair] = 1
                if 'No such contacts encountered' in pication_pi_map.columns.values:
                    pication_pi_map.drop(['No such contacts encountered'], axis=1, inplace=True)

        # This section is to account for additional H-bond strength
        # config.HBOND_DIST_MAX = 3.5
        # config.HBOND_DON_ANGLE_MIN = 140
        # pdb.analyze()
        # for key_res, site in pdb.interaction_sets.items():
        #     # Convert key residue from 3 letter code to 1 letter code
        #     key_res = key_res.replace(key_res.split(':')[0], aa_names.get(key_res.split(':')[0])).split(':')
        #     # Get resID. Convert key residue from format "<resname>:<resid>:<chain>" to "<resname><resid>:<chain>"
        #     key_resID = int(key_res[-1])
        #     key_res = key_res[0] + key_res[-1] + ':' + key_res[1]
        #     # Evaluate all interactions involving this key residue
        #     if len(get_interactions(key_resID, site, 'hbonds')) > 0:
        #         # Record interaction pair to dataframe
        #         for partner_res in get_interactions(key_resID, site, 'hbonds'):
        #             interaction_pair = key_res + '-' + partner_res
        #             hbonds_map.loc[frame_count, interaction_pair] = 2
        #         if 'No such contacts encountered' in hbonds_map.columns.values:
        #             hbonds_map.drop(['No such contacts encountered'], axis=1, inplace=True)

        # If no interactions detected in this frame, give all zeros (np.NaN)
        for interaction_map in [hbonds_map, hydrophobic_map, salt_bridges_map, pication_pi_map]:
            if len(interaction_map.index.values) == 0 or frame_count != interaction_map.index.values[-1]:
                interaction_map.loc[frame_count] = np.NaN

        frame_count += 1
        if frame_count % 5 == 0:
            print('\r   PROGRESS:     ', frame_count, end=' frames', flush=True)
        # if frame_count % 10 == 0:
        #     break
    print('\r   PROGRESS:     ', frame_count, end=' frames (DONE)\n', flush=True)

    # Save results as csv
    for result, interaction_map, interaction_name in zip(saved_results, [hbonds_map, hydrophobic_map, salt_bridges_map, pication_pi_map], ['hbonds', 'hydrophobic', 'salt_bridges', 'pication_pi']):
        # If simulation is not set to analyze intramolecular interactions, remove all intramolecular interactions
        if not arg.intramolecular:
            # Remove every time two of the same seg appears in column name
            regex_statement = ':' + arg.seg1[-1] + '.*?:' + arg.seg1[-1]
            interaction_map = interaction_map.drop(interaction_map.filter(regex=regex_statement).columns, axis=1)
        interaction_map.to_csv(result)
        print('Saved', interaction_name, 'to', result, 'at frame', frame_count)
        # print(interaction_map)
else:
    print('Previously saved results found. Skipping calculations!')
    hbonds_map = pd.read_csv(saved_results[0], index_col=0)
    print('Loaded', saved_results[0], '(last frame:', str(hbonds_map.index.values[-1] + 1) + ')')
    hydrophobic_map = pd.read_csv(saved_results[1], index_col=0)
    print('Loaded', saved_results[1], '(last frame:', str(hydrophobic_map.index.values[-1] + 1) + ')')
    salt_bridges_map = pd.read_csv(saved_results[2], index_col=0)
    print('Loaded', saved_results[2], '(last frame:', str(salt_bridges_map.index.values[-1] + 1) + ')')
    pication_pi_map = pd.read_csv(saved_results[3], index_col=0)
    print('Loaded', saved_results[3], '(last frame:', str(pication_pi_map.index.values[-1] + 1) + ')')
    print('WARNING: User must manually verify frame count stated above is as desired before proceeding!')
    print('         If not, delete all saved files to start fresh calculations.')
    frame_count = str(hbonds_map.index.values[-1] + 1)

# Fill all np.NaN with zeros
for interaction_map in [hbonds_map, hydrophobic_map, salt_bridges_map, pication_pi_map]:
    interaction_map.fillna(0, inplace=True)

# Convert interaction map frame number to equivalent time
for datasets in [hbonds_map, hydrophobic_map, salt_bridges_map, pication_pi_map]:
    datasets.index = [float(int(i) * arg.time_total / len(datasets.index)) for i in datasets.index]

# Set up time course
time = hbonds_map.T.columns.values

################################################################################################

# Set up plots
sns.set_context('talk')
title_size = arg.size
label_size = arg.size
fig1, axes1 = plt.subplots(1, 1, figsize=(19, 3 + len(hbonds_map.columns) * 0.2), num='hbonds')  # rows, columns
fig2, axes2 = plt.subplots(1, 1, figsize=(19, 3 + len(hydrophobic_map.columns) * 0.2), num='hydrophobic')  # rows, columns
fig3, axes3 = plt.subplots(1, 1, figsize=(19, 3 + len(salt_bridges_map.columns) * 0.2), num='salt_bridges')  # rows, columns
fig4, axes4 = plt.subplots(1, 1, figsize=(19, 3 + len(pication_pi_map.columns) * 0.2), num='pication_pi')  # rows, columns

# Plot interactions as heatmap
print('>> Plotting time series of interactions as heatmaps...')
for ax, interaction_map, interaction_name in zip([axes1, axes2, axes3, axes4], [hbonds_map, hydrophobic_map, salt_bridges_map, pication_pi_map], ['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'π/cation-π']):
    if 'No such contacts encountered' in interaction_map.columns.values:
        print('Note: no interactions detected for', interaction_name)
        continue
    # Sort column names by index of key residue
    interaction_map = interaction_map.reindex(sorted(interaction_map.columns, key=lambda x: int(x.split(':')[0][1:])), axis=1)
    # If simulation is not set to analyze intramolecular interactions, remove all intramolecular interactions
    if not arg.intramolecular:
        # Remove every time two of the same seg appears in column name
        regex_statement = ':' + arg.seg1[-1] + '.*?:' + arg.seg1[-1]
        interaction_map = interaction_map.drop(interaction_map.filter(regex=regex_statement).columns, axis=1)
    # Plot data as heatmap
    my_map = sns.heatmap(interaction_map.T, vmin=0, vmax=1, xticklabels=False, yticklabels=1, cbar=False, ax=ax, cmap=matplotlib.colors.ListedColormap(['#ffffff', '#284b63']))
    # Add black frame around heatmap
    for _, spine in my_map.spines.items():
        spine.set_visible(True)
    print('\r   PROGRESS:     ', interaction_name, end='', flush=True)
print('\r   PROGRESS:     ', interaction_name, end=' (DONE)\n', flush=True)

# This section is only used to generate nice looking x labels for the heatmap (heatmap order is categorical, not numeric, so default xticks will be 103, 134, 162,... instead of 100, 200, 300,...)
dummy_ax = Axes(Figure(), [0, 0, 1, 1])
sns.lineplot(x=time, y=range(len(time)), ax=dummy_ax)
dummy_ax.xaxis.set_minor_locator(AutoMinorLocator())
dummy_ax.set(xlim=(0, time[-1]))
major_xticks = list(dummy_ax.get_xticks(minor=False))[:-1]  # Skip last one because it is not plotted for some reason
minor_xticks = list(dummy_ax.get_xticks(minor=True))
major_xtick_locations = []
major_xtick_labels = []
minor_xtick_locations = []

# When I wrote this code, only God and I knew how it worked. Now, only God knows how it works
for t in major_xticks:
    correction_factor = 1 / len(time) * closest(time, t)
    time_index = round(len(time) * t / arg.time_total + correction_factor, 2)
    major_xtick_locations.append(time_index)
    major_xtick_labels.append(int(t))
for t in minor_xticks:
    correction_factor = 1 / len(time) * closest(time, t)
    time_index = round(len(time) * t / arg.time_total + correction_factor, 2)
    minor_xtick_locations.append(time_index)

# Set xticks for interaction map, set rotation of yticks for interaction map to 0
for ax in [axes1, axes2, axes3, axes4]:
    ax.set_xticks(major_xtick_locations)
    ax.set_xticklabels(major_xtick_labels)
    ax.set_xticks(minor_xtick_locations, minor=True)
    ax.yaxis.set_tick_params(rotation=0)

# Set titles and labels of plots
for ax, interaction_name in zip([axes1, axes2, axes3, axes4], ['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'π/cation-π']):
    ax.set_title('Time Series of ' + interaction_name + ' Interactions Between ' + str(arg.seg1_name) + ' & ' + str(arg.seg2_name) + ' - ' + '.'.join(arg.dcd.split('.')[:-1]), y=1.02, fontsize=title_size)
    ax.set_xlabel(arg.x_label, fontsize=label_size)

################################################################################################

# Save figures
print('\nPlots are saved as:')
for figure, interaction_map, interaction_name in zip(['hbonds', 'hydrophobic', 'salt_bridges', 'pication_pi'], [hbonds_map, hydrophobic_map, salt_bridges_map, pication_pi_map], ['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'π/cation-π']):
    plt.figure(figure)
    plt.savefig(os.path.join('figures', file_name + '_' + figure + '_' + name + '.png'), bbox_inches='tight', dpi=200)
    print(os.path.join('figures', file_name + '_' + figure + '_' + name + '.png'))

print(' /\\     /\\')
print('{  `---\'  \\}')
print('{  O   O  }')
print('~~>  V  <~~')
print(' \\  \\|/  /')
print('  `-----\'__')
print('  /     \\  `^\_')
print(' {       }\\ |\\_\\_   W')
print(' |  \\_/  |/ /  \\_\\_( )')
print('  \\__/  /(_E     \\__/')
print('    (  /')
print('     MM')
