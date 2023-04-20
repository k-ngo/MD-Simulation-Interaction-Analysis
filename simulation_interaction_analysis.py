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
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
from datetime import timedelta
import multiprocessing as mp
# matplotlib.use('Agg')
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Configurations
PDB_column_width_format = [(0, 4), (4, 11), (11, 16), (16, 20), (20, 22), (22, 26), (26, 38), (38, 46), (46, 54), (54, 60), (60, 66), (66, 90)]
parser = argparse.ArgumentParser(description='Molecular Interaction Analysis')
parser.add_argument('-b', '--pdb',
                    default=None,
                    dest='pdb', action='store',
                    help='.pdb file for single-frame analysis')
parser.add_argument('-p', '--psf',
                    default=glob.glob('*.psf')[0],
                    dest='psf', action='store',
                    help='.psf file containing protein structural information')
parser.add_argument('-d', '--dcd',
                    default=glob.glob('*.dcd')[0],
                    dest='dcd', action='store',
                    help='.dcd file containing simulation trajectory (any trajectory format will also work)')
parser.add_argument('-s1',
                    dest='sel1', action='store',
                    help='First segment/chain/subunit to consider for analysis (follows VMD format for selection)')
parser.add_argument('-s2',
                    dest='sel2', action='store',
                    help='Second segment/chain/subunit to consider for analysis (follows VMD format for selection)')
parser.add_argument('-s1n',
                    default='SEL1',
                    dest='sel1_name', action='store',
                    help='Name of first segment/chain/subunit to consider for analysis (customized by user)')
parser.add_argument('-s2n',
                    default='SEL2',
                    dest='sel2_name', action='store',
                    help='Name of second segment/chain/subunit to consider for analysis (customized by user)')
parser.add_argument('-s1x',
                    default='',
                    dest='sel1_exclude', action='store',
                    help='Exclude this VMD selection from the first segment/chain/subunit')
parser.add_argument('-s2x',
                    default='',
                    dest='sel2_exclude', action='store',
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
parser.add_argument('-m', '--minimum',
                    default=5,
                    dest='minimum', action='store', type=float,
                    help='Only display interactions that occur for at least this percentage of the total simulation time')
parser.add_argument('-n', '--threads',
                    default=None,
                    dest='num_threads', action='store', type=int,
                    help='Number of threads to use for parallel processing')
parser.add_argument('--split',
                    default=1,
                    dest='split', action='store', type=int,
                    help='split each plot into # of smaller plots covering different time periods, useful for long simulations')
parser.add_argument('--skipcommand',
                    dest='skip_command', action='store_true',
                    help='if toggled, skip running VMD commands to generate input data, only set if the script has already been ran at least once')
parser.add_argument('--labelsize',
                    default=15,
                    dest='size', action='store', type=float,
                    help='label font size (default = 15)')
arg = parser.parse_args()

# Conversion of 3-letter amino acid code to 1-letter code
aa_names = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
            'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
            'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
            'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'SAP': '*', 'POP': '#'}

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


def get_interactions(site, type):
    """Return residue numbers of residues involved in specified type of interactions"""
    if type == 'hbonds':
        # return list(set([get_res_properties(i) for i in site.hbonds_ldon if abs(resID - i.resnr) != 4] + [get_res_properties(i) for i in site.hbonds_pdon if abs(resID - i.resnr) != 4]))
        return list(set([get_res_properties(i) for i in site.hbonds_ldon] + [get_res_properties(i) for i in site.hbonds_pdon]))
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


def record_interactions(interaction_list, frame, key_res, interaction_map, temp_interaction_map):
    """Record interactions to dataframe"""
    if len(interaction_list) > 0:
        # Record interaction pair to dataframe
        for partner_res in interaction_list:
            # Check if key_res and partner_res are the same residue
            if key_res != partner_res:
                # Record interaction pair to dataframe, place residue in selection 1 first
                if key_res[1:] in interacting_sel1:
                    interaction_pair = key_res + '-' + partner_res
                    reversed_interaction_pair = partner_res + '-' + key_res
                elif partner_res[1:] in interacting_sel1:
                    interaction_pair = partner_res + '-' + key_res
                    reversed_interaction_pair = key_res + '-' + partner_res
                else:
                    continue
                # Check of duplicate interaction pair like AB-CD and CD-AB
                if reversed_interaction_pair not in temp_interaction_map.columns.values and reversed_interaction_pair not in interaction_map.columns.values:
                    temp_interaction_map.loc[frame, interaction_pair] = 1


def closest(input_list, k):
    """Find the closest number to k in list"""
    input_list = np.asarray(input_list)
    index = (np.abs(input_list - k)).argmin()
    return index


def letter_to_number(input):
    """Convert letter to number in residue number format for sorting"""
    # Example: convert 'S765:C-S240:A' to '765671024065' (S765:C = 765 67, S240:A = 240 65)
    resid1 = input.split('-')[0][1:].replace(':', '')
    resid1 = str(resid1[:-1]) + str(ord(resid1[-1]))
    resid2 = input.split('-')[1][1:].replace(':', '')
    resid2 = str(resid2[:-1]) + str(ord(resid2[-1]))
    # Check for special characters in residue name
    if input[0].isalpha():
        # If resname is protein, add 1,000,000 to second residue number to ensure correct sorting
        return int(str(resid1) + str((1000000 + int(resid2))))
    else:
        # If resname is not protein ('*' or '#'), also add 1,000,000 to first residue number to makea sure they stay at the bottom
        return int(str((1000000 + int(resid1))) + str((1000000 + int(resid2))))


def analyze_frame(i):
    global frame_count
    # Read frame
    try:
        frame = pd.read_fwf(frame_pdb + str(i) + '.pdb', header=None, colspecs=PDB_column_width_format, skiprows=1, skipfooter=1)
    except:
        for interaction_map in [hbonds_map, hydrophobic_map, salt_bridges_map, pi_stacking_map, pi_cation_map]:
            interaction_map.loc[i] = np.NaN
    # Name columns
    frame.columns = ['Atom', 'Index', 'Type', 'Residue', 'Chain', 'ResNum', 'X', 'Y', 'Z', 'Occupancy', 'TempFactor', 'Segment']
    # Change water numbering to all integers
    frame.loc[frame['Residue'] == 'TIP', 'ResNum'] = frame['Index']
    # Convert float to int for select columns
    frame['ResNum'] = frame['ResNum'].astype(int)
    # frame.loc[frame['Residue'] != 'TIP', 'SeqNum'] += 401
    frame['Index'] = frame['Index'].astype(int)
    # Convert "HSD" to "HIS"
    frame.loc[frame['Residue'] == 'HSD', 'Residue'] = 'HIS'
    # Fix chain name
    frame['Chain'] = frame['Segment'].str[-1]
    frame.loc[frame['Chain'] == '3', 'Chain'] = 'W'
    # Convert atom column to 'HETATM' for analysis
    frame['Atom'] = 'HETATM'
    # frame.loc[(frame['ResNum'] == res) & (frame['Segment'] == segname), 'Atom'] = 'HETATM'
    # config.PEPTIDES = ['X']
    # Change segment column to element symbol
    if not arg.pdb:
        frame['Segment'] = frame['Type'].str[0]
    # DEBUG - Write selection in each frame to file
    with open(os.path.join(debug_folder, 'example_selection_frame_' + str(i) + '.txt'), 'w+') as f:
        f.write(frame.to_string())
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
    # Initialize dataframe for storing interactions
    temp_hbonds_map = pd.DataFrame(columns=['No such contacts encountered'])
    temp_hydrophobic_map = pd.DataFrame(columns=['No such contacts encountered'])
    temp_salt_bridges_map = pd.DataFrame(columns=['No such contacts encountered'])
    temp_pi_stacking_map = pd.DataFrame(columns=['No such contacts encountered'])
    temp_pi_cation_map = pd.DataFrame(columns=['No such contacts encountered'])
    # for ligand in pdb.ligands:
    #     print(ligand)
    # longnames = [x.longname for x in pdb.ligands]
    # bsids = [":".join([x.hetid, x.chain, str(x.position)]) for x in pdb.ligands]
    # print('*', longnames)
    # print('!', bsids)
    # for bsid in bsids:
    #     interactions = pdb.interaction_sets[bsid]
    #     print(interactions.saltbridge_lneg, interactions.saltbridge_pneg)
    for key_res, site in pdb.interaction_sets.items():
        # Convert key residue from 3 letter code to 1 letter code
        try:
            key_res = key_res.replace(key_res.split(':')[0], aa_names.get(key_res.split(':')[0])).split(':')
        except:
            key_res = key_res.replace(key_res.split(':')[0], '*').split(':')
        # Convert key residue from format "<resname>:<resid>:<chain>" to "<resname><resid>:<chain>"
        key_res = key_res[0] + key_res[-1] + ':' + key_res[1]
        # Evaluate all interactions involving this key residue
        record_interactions(get_interactions(site, 'hbonds'), i, key_res, hbonds_map, temp_hbonds_map)
        record_interactions(get_interactions(site, 'hydrophobic_contacts'), i, key_res, hydrophobic_map, temp_hydrophobic_map)
        record_interactions(get_interactions(site, 'salt_bridges'), i, key_res, salt_bridges_map, temp_salt_bridges_map)
        record_interactions(get_interactions(site, 'pi_stacking'), i, key_res, pi_stacking_map, temp_pi_stacking_map)
        record_interactions(get_interactions(site, 'pi_cation'), i, key_res, pi_cation_map, temp_pi_cation_map)
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
    for interaction_map in [temp_hbonds_map, temp_hydrophobic_map, temp_salt_bridges_map, temp_pi_stacking_map, temp_pi_cation_map]:
        if len(interaction_map.index.values) == 0 or i != interaction_map.index.values[-1]:
            interaction_map.loc[i] = np.NaN
    return temp_hbonds_map, temp_hydrophobic_map, temp_salt_bridges_map, temp_pi_stacking_map, temp_pi_cation_map


def collect_result(results):
    global frame_count, hbonds_map, hydrophobic_map, salt_bridges_map, pi_stacking_map, pi_cation_map
    temp_hbonds_map, temp_hydrophobic_map, temp_salt_bridges_map, temp_pi_stacking_map, temp_pi_cation_map = results
    # Collect results from each frame
    hbonds_map = pd.concat([hbonds_map, temp_hbonds_map])
    hydrophobic_map = pd.concat([hydrophobic_map, temp_hydrophobic_map])
    salt_bridges_map = pd.concat([salt_bridges_map, temp_salt_bridges_map])
    pi_stacking_map = pd.concat([pi_stacking_map, temp_pi_stacking_map])
    pi_cation_map = pd.concat([pi_cation_map, temp_pi_cation_map])
    # Print progress
    frame_count += 1
    time_elapsed = time.time() - start
    avg_time_per_frame = time_elapsed / frame_count
    frames_left = num_frames - frame_count
    print('\r   PROGRESS:     ', str(frame_count) + '/' + str(num_frames), 'frames',
          '| TIME ELAPSED: ', timedelta(seconds=time_elapsed),
          '| COMPLETE IN:  ', timedelta(seconds=frames_left * avg_time_per_frame), end=' ', flush=True)


def separate_into_inter_intrachain(interaction_map):
    """Separate interaction map results into intra- and interchain interactions"""
    if 'No such contacts encountered' not in interaction_map.columns.values:
        # Create regex pattern to find the first character after the first colon
        regex_pattern = r':(.)'  # 1 character immediately after ":"
        intrachain_interaction_pairs = []
        for interaction_pair in interaction_map.columns.values:
            # (Find intra-chain occurrences) Check if the first character after the first colon matches the second character after the second colon
            if re.findall(regex_pattern, interaction_pair)[0] == re.findall(regex_pattern, interaction_pair)[1]:
                intrachain_interaction_pairs.append(interaction_pair)
        # Save intrachain and interchain interactions to separate dataframes
        interchain_interaction_map = interaction_map.drop(intrachain_interaction_pairs, axis=1)
        intrachain_interaction_map = interaction_map.drop([col for col in interaction_map.columns if col not in intrachain_interaction_pairs], axis=1)
        return interchain_interaction_map, intrachain_interaction_map
    else:
        return null_map, null_map


def plot_interactions(interaction_map, interaction_name, result):
    # Set up figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(15, 3 + len(interaction_map.columns) * 0.2), num='timeseries')  # rows, columns
    fig2, ax2 = plt.subplots(1, 1, figsize=(4 + len(interaction_map.columns) * 0.2, 2.5 + len(interaction_map.columns) * 0.2), num='percentage')  # rows, columns
    print('  ', interaction_name, end='\n')
    # Sort column names by residue number and chain ID
    interaction_map = interaction_map.reindex(sorted(interaction_map.columns, key=lambda x: letter_to_number(x)), axis=1)
    #############################################################################################
    # Prepare data for plotting
    #############################################################################################
    # Transpose the map
    interaction_map = interaction_map.T
    #############################################################################################
    # Plot time series of interactions as a heatmap
    #############################################################################################
    time_series_plot = sns.heatmap(interaction_map, vmin=0, vmax=1, xticklabels=False, yticklabels=1, linewidths=0.2, cbar=False,
                                   ax=ax1, cmap=matplotlib.colors.ListedColormap(['#1C1427', '#52D681']), annot_kws={'size': 25 / np.sqrt(len(interaction_map.columns))})
    # Add black frame around heatmap
    for _, spine in time_series_plot.spines.items():
        spine.set_visible(True)
    #############################################################################################
    # Plot the number of interactions for each interacting pair as a percentage of total frames
    #############################################################################################
    interaction_map['Total'] = interaction_map.sum(axis=1, numeric_only=True)
    percentage_interaction_map = pd.DataFrame()
    # Loop through each row and calculate the percentage of frames that interaction occurs
    for index, row in interaction_map.iterrows():
        first_res, second_res = index.split('-')[0], index.split('-')[1]
        percentage_interaction_map.loc[first_res, second_res] = round(interaction_map['Total'][index] / (len(interaction_map.columns) - 1) * 100, 1)  # -1 here to exclude the 'Total' column from the count
    # Extract the row and column names of each cell that contains a value:
    with open(os.path.join(saved_results_folder, interaction_name.replace(' ', '_') + '.txt'), 'w+') as f:
        for index, row in percentage_interaction_map.iterrows():
            for column in percentage_interaction_map.columns:
                if not np.isnan(percentage_interaction_map.loc[index, column]):
                    f.write(str(index) + '-' + str(column) + '\n')
                    # print('Percentage of frames that', index, 'forms', interaction_name, 'with', column, '=', percentage_interaction_map.loc[index, column], '%')
    # # Transpose, then sort column names by index of key residue, then transpose again
    # percentage_interaction_map = percentage_interaction_map.T
    # percentage_interaction_map = percentage_interaction_map.reindex(sorted(percentage_interaction_map.index, key=lambda x: int(x.split(':')[0][1:])), axis=0)
    # Replace all NaN with zeroes
    percentage_interaction_map.fillna(0, inplace=True)
    # Add annotations to display percentage in each cell
    annotations = percentage_interaction_map.values.astype(str)
    annotations[annotations == '0.0'] = ''
    # Save data
    percentage_interaction_map.to_csv(result)
    # Plot the data as an interaction matrix
    percentage_plot = sns.heatmap(percentage_interaction_map, xticklabels=1, yticklabels=1, vmin=0, vmax=100, linewidths=0.2,
                                  cmap=sns.color_palette('magma', as_cmap=True), cbar=False, annot=annotations, fmt='s', annot_kws={'size': 25 / np.sqrt(len(percentage_interaction_map.columns))}, ax=ax2)
    # Add black frame around heatmap
    for _, spine in percentage_plot.spines.items():
        spine.set_visible(True)
    # for text in percentage_plot.texts:
    #     if len(text.get_text()) > 0:
    #         text.set_text(text.get_text() + '%')
    # Generate color bar
    # axins = inset_axes(ax2,
    #                    width=0.3,
    #                    height=5,
    #                    loc='upper right',
    #                    bbox_to_anchor=(0.04, 0, 1, 1),
    #                    # (x0, y0, width, height) where (x0,y0) are the lower left corner coordinates of the bounding box
    #                    bbox_transform=ax2.transAxes,
    #                    borderpad=0)
    # cb1 = matplotlib.colorbar.ColorbarBase(axins, cmap=sns.color_palette('magma', as_cmap=True),
    #                                        norm=matplotlib.colors.Normalize(vmin=0, vmax=100),
    #                                        orientation='vertical')
    # cb1.set_label('% of time contacts are formed')

    # This section is only used to generate nice looking x labels for the time series heatmap (heatmap order is categorical, not numeric, so default xticks will be 103, 134, 162,... instead of 100, 200, 300,...)
    dummy_ax = Axes(Figure(), [0, 0, 1, 1])
    sns.lineplot(x=time, y=range(len(time)), ax=dummy_ax)
    dummy_ax.xaxis.set_minor_locator(AutoMinorLocator())
    dummy_ax.set(xlim=(0, time[-1]))
    major_xticks = list(dummy_ax.get_xticks(minor=False))[:-1]  # Skip last one because it is not plotted for some reason
    minor_xticks = list(dummy_ax.get_xticks(minor=True))
    major_xtick_locations = []
    major_xtick_labels = []
    minor_xtick_locations = []

    # Make sure that tick locations are correct
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

    # Set xticks for time series interaction map, set rotation of y ticks to 0
    ax1.set_xticks(major_xtick_locations)
    ax1.set_xticklabels(major_xtick_labels)
    ax1.set_xticks(minor_xtick_locations, minor=True)
    ax1.yaxis.set_tick_params(rotation=0)

    # Tilt xticks for percentage interaction map, set rotation of y ticks to 0
    ax2.set_xticklabels(ax2.get_xticklabels(),
                        rotation=45,
                        horizontalalignment='right')
    ax2.yaxis.set_tick_params(rotation=0)

    # Plot time series interaction map
    plt.figure('timeseries')
    plt.tight_layout()
    if 'Interchain' in interaction_name:
        plt.savefig(os.path.join(figures_folder, 'interchain', 'timeseries_' + interaction_name.replace(' ', '_') + '.png'), bbox_inches='tight', dpi=300)
    elif 'Intrachain' in interaction_name:
        plt.savefig(os.path.join(figures_folder, 'intrachain', 'timeseries_' + interaction_name.replace(' ', '_') + '.png'), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(figures_folder, 'timeseries_' + interaction_name.replace(' ', '_') + '.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Plot percentage interaction map
    plt.figure('percentage')
    plt.tight_layout()
    if 'Interchain' in interaction_name:
        plt.savefig(os.path.join(figures_folder, 'interchain', 'percentage_' + interaction_name.replace(' ', '_') + '.png'), bbox_inches='tight', dpi=300)
    elif 'Intrachain' in interaction_name:
        plt.savefig(os.path.join(figures_folder, 'intrachain', 'percentage_' + interaction_name.replace(' ', '_') + '.png'), bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(figures_folder, 'percentage_' + interaction_name.replace(' ', '_') + '.png'), bbox_inches='tight', dpi=300)
    plt.close()


# Automatically determine input file name if given wildcard as input - will take first result that appears as input
if arg.psf.split('.')[0] == '*' and not arg.pdb:
    arg.psf = glob.glob('*.' + arg.psf.split('.')[-1])[0]
if arg.dcd.split('.')[0] == '*' and not arg.pdb:
    arg.dcd = glob.glob('*.' + arg.dcd.split('.')[-1])[0]

# If there are atoms to be excluded, generate VMD commands to do so
sel1_exclude = ''
sel2_exclude = ''
if arg.sel1_exclude:
    sel1_exclude = ' and not (' + str(arg.sel1_exclude) + ')'
if arg.sel2_exclude:
    sel2_exclude = ' and not (' + str(arg.sel2_exclude) + ')'

# Print input information
if arg.pdb:
    print('PDB       |  ', arg.pdb)
else:
    print('PSF       |  ', arg.psf)
    print('DCD       |  ', arg.dcd)
print('1st Sel   |  ', arg.sel1_name, '-', arg.sel1)
if arg.sel1_exclude:
    print('              excluding', str(arg.sel1_exclude))
print('2nd Sel   |  ', arg.sel2_name, '-', arg.sel2)
if arg.sel2_exclude:
    print('              excluding', str(arg.sel2_exclude))
if arg.num_threads:
    print('# threads |  ', arg.num_threads, '(max = ' + str(mp.cpu_count()) + ')')

# Create folders to store data and output
name = str(arg.sel1_name).replace(' ', '-') + '_' + str(arg.sel2_name).replace(' ', '-')
if arg.pdb:
    file_name = '.'.join(arg.pdb.split('.')[:-1])
else:
    file_name = '.'.join(arg.dcd.split('.')[:-1])

temp_pdb_folder = 'temp_pdb_' + file_name + '_' + name
debug_folder = 'debug_' + file_name + '_' + name
saved_results_folder = 'saved_results_' + file_name + '_' + name
figures_folder = 'figures_' + file_name + '_' + name
os.makedirs(temp_pdb_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)
os.makedirs(saved_results_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)
os.makedirs(os.path.join(figures_folder, 'interchain'), exist_ok=True)
os.makedirs(os.path.join(figures_folder, 'intrachain'), exist_ok=True)
os.makedirs(os.path.join(saved_results_folder, 'interchain'), exist_ok=True)
os.makedirs(os.path.join(saved_results_folder, 'intrachain'), exist_ok=True)
frame_pdb = os.path.join(temp_pdb_folder, str(arg.sel1_name).replace(' ', '-') + '_' + str(arg.sel2_name).replace(' ', '-') + '_')

# Check required files
for script in ['prot_center.tcl']:
    if not os.path.exists(script):
        print('ERROR: Required script', script, 'not found in current directory.')
        exit(1)

#############################################################################################
# Load simulation trajectory and extract data
#############################################################################################
vmd_cmd_file = file_name + '_vmd_cmd.tcl'
print('\n#######################################################################')
if not arg.skip_command:
    with open(vmd_cmd_file, 'w+') as f:
        # Load input files
        if arg.pdb:
            # Load PDB file
            f.write('mol new ' + arg.pdb + '\n\n')
        else:
            # Load trajectory files
            f.write('mol new ' + arg.psf + ' type psf first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all\n')
            f.write('mol addfile ' + arg.dcd + ' type ' + arg.dcd.split('.')[-1] + ' first 0 last ' + str(arg.end_frame) + ' step ' + str(arg.step) + ' filebonds 1 autobonds 1 waitfor all\n')
            # Center protein
            f.write('source prot_center.tcl\n\n')

        # Obtain number of frames
        f.write('set nframes "' + os.path.join(temp_pdb_folder, name + '_nframes.txt') + '"\n')
        f.write('set outframes [open ${nframes} w]\n')
        f.write('puts $outframes "[molinfo top get numframes]"\n')
        f.write('close $outframes\n\n')

        # Set interacting segment names
        f.write('set sel1 "' + str(arg.sel1) + '"\n')
        f.write('set sel2 "' + str(arg.sel2) + '"\n')

        # Initialize files to store data
        f.write('set outputname_sel1_resid "' + os.path.join(temp_pdb_folder, name + '_sel1_resid.dat') + '"\n')
        f.write('set outputname_sel2_resid "' + os.path.join(temp_pdb_folder, name + '_sel2_resid.dat') + '"\n')
        f.write('set outputname_sel1_segment "' + os.path.join(temp_pdb_folder, name + '_sel1_segment.dat') + '"\n')
        f.write('set outputname_sel2_segment "' + os.path.join(temp_pdb_folder, name + '_sel2_segment.dat') + '"\n')

        f.write('set out1 [open ${outputname_sel1_resid} w]\n')
        f.write('set out2 [open ${outputname_sel2_resid} w]\n')
        f.write('set out3 [open ${outputname_sel1_segment} w]\n')
        f.write('set out4 [open ${outputname_sel2_segment} w]\n\n')

        # Obtain atoms from sel1 and sel2, then write to files
        f.write('set interacting_sel1 [atomselect top "($sel1' + sel1_exclude + ')"]\n')
        f.write('set interacting_sel2 [atomselect top "($sel2' + sel1_exclude + ')"]\n')
        f.write('puts $out1 "[$interacting_sel1 get resid]"\n')
        f.write('puts $out2 "[$interacting_sel2 get resid]"\n')
        if arg.pdb:
            f.write('puts $out3 "[$interacting_sel1 get chain]"\n')
            f.write('puts $out4 "[$interacting_sel2 get chain]"\n\n')
        else:
            f.write('puts $out3 "[$interacting_sel1 get segname]"\n')
            f.write('puts $out4 "[$interacting_sel2 get segname]"\n\n')

        # Loop through each frame to obtain atoms from sel1 and sel2 that are within max interaction distance (defined by config.BS_DIST)
        f.write('set nf [molinfo top get numframes]\n')
        f.write('for {set f 0} {$f < $nf} {incr f} {\n')
        # ' + str(config.BS_DIST) + '
        f.write('set interacting_sel1_sel2 [atomselect top "(($sel1' + sel1_exclude + ') and same residue as within ' + str(config.BS_DIST) + ' of ($sel2' + sel2_exclude + ')) or (($sel2' + sel2_exclude + ') and same residue as within ' + str(config.BS_DIST) + ' of ($sel1' + sel1_exclude + '))" frame $f]\n')
        f.write('$interacting_sel1_sel2 frame $f\n')
        f.write('$interacting_sel1_sel2 writepdb ' + frame_pdb + '$f.pdb }\n\n')

        f.write('close $out1\n')
        f.write('close $out2\n')
        f.write('close $out3\n')
        f.write('close $out4\n')

        f.write('exit')

    # sp.run(['/bin/bash', '-i', '-c', 'vmd -dispdev text -e ' + vmd_cmd_file], start_new_session=True, shell=True)
    # cmd = ['vmd', '-dispdev', 'text', '-e', vmd_cmd_file,]
    # sp.Popen(cmd, stdin=sp.PIPE)
    sp.call(['/bin/bash', '-c', '-i', 'vmd -dispdev text -e ' + vmd_cmd_file], stdin=sp.PIPE)

# Read IDs and segname of residues from sel1 that can possibly interact with sel2
with open(os.path.join(temp_pdb_folder, name + '_sel1_resid.dat')) as f:
    interacting_sel1_resid = [list(map(int, i.split())) for i in f.read().splitlines()][0]
with open(os.path.join(temp_pdb_folder, name + '_sel1_segment.dat')) as f:
    interacting_sel1_segname = [i.split() for i in f.read().splitlines()][0]

# Combine IDs and segname, removing any duplicates
interacting_sel1 = list(set([str(i) + ':' + str(j[-1]) for i, j in zip(interacting_sel1_resid, interacting_sel1_segname)]))

# Obtain number of frames
with open(os.path.join(temp_pdb_folder, name + '_nframes.txt')) as f:
    num_frames = int(f.read().splitlines()[0])

# Set up interaction maps
# ['H-bonds', 'Hydrophobic', 'Water bridges', 'Salt bridges', 'Pi_cation-pi']
hbonds_map = pd.DataFrame(columns=['No such contacts encountered'])
hydrophobic_map = pd.DataFrame(columns=['No such contacts encountered'])
salt_bridges_map = pd.DataFrame(columns=['No such contacts encountered'])
pi_stacking_map = pd.DataFrame(columns=['No such contacts encountered'])
pi_cation_map = pd.DataFrame(columns=['No such contacts encountered'])
null_map = pd.DataFrame(columns=['No such contacts encountered'])

#############################################################################################
# Analyze interactions throughout simulation trajectory
#############################################################################################
frame_count = 0
print('#######################################################################')
print('\n>> Analyzing molecular interactions')
# Look for previous results first, if not saved then proceed with calculations
saved_results = [os.path.join(saved_results_folder, figure + '_' + name + '.csv') for figure in ['hbonds', 'hydrophobic', 'salt_bridges', 'pi_stacking', 'pi_cation']]
saved_results_percentage = [os.path.join(saved_results_folder, figure + '_' + name + '.csv') for figure in ['hbonds_percentage', 'hydrophobic_percentage', 'salt_bridges_percentage', 'pi_stacking_percentage', 'pi_cation_percentage']]
interchain_saved_results = [os.path.join(saved_results_folder, 'interchain', figure + '_' + name + '.csv') for figure in ['hbonds', 'hydrophobic', 'salt_bridges', 'pi_stacking', 'pi_cation']]
interchain_saved_results_percentage = [os.path.join(saved_results_folder, 'interchain', figure + '_' + name + '.csv') for figure in ['hbonds_percentage', 'hydrophobic_percentage', 'salt_bridges_percentage', 'pi_stacking_percentage', 'pi_cation_percentage']]
intrachain_saved_results = [os.path.join(saved_results_folder, 'intrachain', figure + '_' + name + '.csv') for figure in ['hbonds', 'hydrophobic', 'salt_bridges', 'pi_stacking', 'pi_cation']]
intrachain_saved_results_percentage = [os.path.join(saved_results_folder, 'intrachain', figure + '_' + name + '.csv') for figure in ['hbonds_percentage', 'hydrophobic_percentage', 'salt_bridges_percentage', 'pi_stacking_percentage', 'pi_cation_percentage']]
start = time.time()
if not all(os.path.exists(result) for result in saved_results + interchain_saved_results + intrachain_saved_results):
    # Loop through each frame to analyze interactions
    if arg.num_threads:
        # Multiprocessing
        pool = mp.Pool(arg.num_threads)
        for i in range(num_frames):
            # Check for emergency stop signal
            if os.path.exists('STOP'):
                print('>>>>>>>>>>>>>>>>>>>>>> STOPPING <<<<<<<<<<<<<<<<<<<<<<')
                exit(1)
            pool.apply_async(analyze_frame, args=(i,), callback=collect_result)
        pool.close()
        pool.join()
    else:
        # Single processing
        for i in range(num_frames):
            # Check for emergency stop signal
            if os.path.exists('STOP'):
                print('>>>>>>>>>>>>>>>>>>>>>> STOPPING <<<<<<<<<<<<<<<<<<<<<<')
                exit(1)
            collect_result(analyze_frame(i))
    print('\r   PROGRESS:     ', frame_count, end=' frames (DONE)\n', flush=True)
    # Sort index
    hbonds_map.sort_index(inplace=True)
    hydrophobic_map.sort_index(inplace=True)
    salt_bridges_map.sort_index(inplace=True)
    pi_stacking_map.sort_index(inplace=True)
    pi_cation_map.sort_index(inplace=True)
    # Check if any interactions were found
    for result in [hbonds_map, hydrophobic_map, salt_bridges_map, pi_stacking_map, pi_cation_map]:
        # If so drop the 'No such contacts encountered' column
        if len(result.columns.values) > 1:
            result.drop(['No such contacts encountered'], axis=1, inplace=True)
    # Separate results into intra and interchain interactions
    interchain_hbonds_map, intrachain_hbonds_map = separate_into_inter_intrachain(hbonds_map)
    interchain_hydrophobic_map, intrachain_hydrophobic_map = separate_into_inter_intrachain(hydrophobic_map)
    interchain_salt_bridges_map, intrachain_salt_bridges_map = separate_into_inter_intrachain(salt_bridges_map)
    interchain_pi_stacking_map, intrachain_pi_stacking_map = separate_into_inter_intrachain(pi_stacking_map)
    interchain_pi_cation_map, intrachain_pi_cation_map = separate_into_inter_intrachain(pi_cation_map)

    # Save results as csv
    for result, interaction_map in zip(saved_results + interchain_saved_results + intrachain_saved_results,
                                       [hbonds_map, hydrophobic_map, salt_bridges_map, pi_stacking_map, pi_cation_map,
                                        interchain_hbonds_map, interchain_hydrophobic_map, interchain_salt_bridges_map, interchain_pi_stacking_map, interchain_pi_cation_map,
                                        intrachain_hbonds_map, intrachain_hydrophobic_map, intrachain_salt_bridges_map, intrachain_pi_stacking_map, intrachain_pi_cation_map]):
        # Prepare data to be saved
        if 'No such contacts encountered' not in interaction_map.columns.values:
            # Sort column names by residue number and chain ID
            interaction_map = interaction_map.reindex(sorted(interaction_map.columns, key=lambda x: letter_to_number(x)), axis=1)
            # Transpose the map
            interaction_map = interaction_map.T
            interaction_map['Total'] = interaction_map.sum(axis=1, numeric_only=True)
            interaction_map['%'] = round(interaction_map['Total'] / (len(interaction_map.columns) - 1) * 100, 1)
            # Hide all residues pairs that fail to form specified contact frequency over simulation trajectory when plotting
            minimum_contact_percentage = arg.minimum
            interaction_map = interaction_map[interaction_map['%'] >= minimum_contact_percentage]
            interaction_map = interaction_map.drop(columns=['Total', '%'])
            interaction_map = interaction_map.T
            # Save interaction map to csv
            if len(interaction_map.columns.values) > 0:
                interaction_map.to_csv(result)
            else:
                null_map.to_csv(result)
        else:
            null_map.to_csv(result)
    print('   Saved results to', saved_results_folder)
else:
    print('******************************************************')
    print('Previously saved results found. Skipping calculations!')
    print('NOTE: Delete all saved files to start fresh calculations.')
    print('******************************************************')

# Load previously saved results
# Combined results
hbonds_map = pd.read_csv(saved_results[0], index_col=0)
hydrophobic_map = pd.read_csv(saved_results[1], index_col=0)
salt_bridges_map = pd.read_csv(saved_results[2], index_col=0)
pi_stacking_map = pd.read_csv(saved_results[3], index_col=0)
pi_cation_map = pd.read_csv(saved_results[4], index_col=0)
# Interchain results
interchain_hbonds_map = pd.read_csv(interchain_saved_results[0], index_col=0)
interchain_hydrophobic_map = pd.read_csv(interchain_saved_results[1], index_col=0)
interchain_salt_bridges_map = pd.read_csv(interchain_saved_results[2], index_col=0)
interchain_pi_stacking_map = pd.read_csv(interchain_saved_results[3], index_col=0)
interchain_pi_cation_map = pd.read_csv(interchain_saved_results[4], index_col=0)
# Intrachain results
intrachain_hbonds_map = pd.read_csv(intrachain_saved_results[0], index_col=0)
intrachain_hydrophobic_map = pd.read_csv(intrachain_saved_results[1], index_col=0)
intrachain_salt_bridges_map = pd.read_csv(intrachain_saved_results[2], index_col=0)
intrachain_pi_stacking_map = pd.read_csv(intrachain_saved_results[3], index_col=0)
intrachain_pi_cation_map = pd.read_csv(intrachain_saved_results[4], index_col=0)

# Process interaction maps
for interaction_map in [hbonds_map, hydrophobic_map, salt_bridges_map, pi_stacking_map, pi_cation_map,
                        interchain_hbonds_map, interchain_hydrophobic_map, interchain_salt_bridges_map, interchain_pi_stacking_map, interchain_pi_cation_map,
                        intrachain_hbonds_map, intrachain_hydrophobic_map, intrachain_salt_bridges_map, intrachain_pi_stacking_map, intrachain_pi_cation_map]:
    # Fill all np.NaN with zeros
    interaction_map.fillna(0, inplace=True)
    # Convert interaction map frame number to equivalent time
    interaction_map.index = [float(int(i) * arg.time_total / len(interaction_map.index)) for i in interaction_map.index]

# Set up time course
time = hbonds_map.T.columns.values

#############################################################################################
# Plot time series of interactions as heatmap
#############################################################################################
# Set up plots
sns.set_context('talk')
title_size = arg.size
label_size = arg.size

# Plot time series and percentages of interactions as heatmaps
print('>> Plotting interactions as heatmaps')
no_interactions = []
if arg.num_threads:
    pool = mp.Pool(arg.num_threads)
for interaction_map, interaction_name, result in zip([hbonds_map, hydrophobic_map, salt_bridges_map, pi_stacking_map, pi_cation_map,
                                                      interchain_hbonds_map, interchain_hydrophobic_map, interchain_salt_bridges_map, interchain_pi_stacking_map, interchain_pi_cation_map,
                                                      intrachain_hbonds_map, intrachain_hydrophobic_map, intrachain_salt_bridges_map, intrachain_pi_stacking_map, intrachain_pi_cation_map],
                                                      ['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'Pi Stacking', 'Pi-Cation',
                                                       'Interchain Hydrogen Bonding', 'Interchain Hydrophobic', 'Interchain Salt Bridges', 'Interchain Pi Stacking', 'Interchain Pi-Cation',
                                                       'Intrachain Hydrogen Bonding', 'Intrachain Hydrophobic', 'Intrachain Salt Bridges', 'Intrachain Pi Stacking', 'Intrachain Pi-Cation'],
                                                      saved_results_percentage + interchain_saved_results_percentage + intrachain_saved_results_percentage):
    if 'No such contacts encountered' in interaction_map.columns.values:
        no_interactions.append(interaction_name)
        continue
    if arg.num_threads:
        # Multiprocessing
        pool.apply_async(plot_interactions, args=(interaction_map, interaction_name, result,))
    else:
        # Single processing
        plot_interactions(interaction_map, interaction_name, result)
if arg.num_threads:
    pool.close()
    pool.join()

if no_interactions:
    print('\n   NOTE: No interactions detected for', no_interactions)

# Set titles and labels of plots
# for ax, interaction_name in zip([axes1, axes2, axes3, axes4], ['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'Pi_cation-pi']):
#     ax.set_title('Time Series of ' + interaction_name + ' Interactions Between ' + str(arg.sel1_name) + ' & ' + str(arg.sel2_name) + ' - ' + '.'.join(arg.dcd.split('.')[:-1]), y=1, fontsize=title_size)

# for ax, interaction_name in zip([axes5, axes6, axes7, axes8], ['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'Pi_cation-pi']):
#     ax.set_title('Percentage of ' + interaction_name + ' Interactions Between ' + str(arg.sel1_name) + ' & ' + str(arg.sel2_name) + '\n' + '.'.join(arg.dcd.split('.')[:-1]), y=1.01, fontsize=title_size)

################################################################################################
# Generate report
with open(os.path.join(saved_results_folder, 'top_contacts.csv'), 'w+') as f:
    sum_contacts = [0, 0, 0, 0, 0]
    sum_top_contacts = [0, 0, 0, 0, 0]
    sum_remaining_contacts = [0, 0, 0, 0, 0]
    f.write('Type,Pair,% of time in contact\n')
    for name, num in zip(['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'Pi Stacking', 'Pi-Cation'], range(5)):
        if name not in no_interactions:
            # Load contact map
            try:
                contact_map = pd.read_csv(saved_results_percentage[num], index_col=0)
            except:
                continue
            for pair in contact_map.stack().nlargest(100).index:
                sum_contacts[num] += contact_map.loc[pair]
                if round(contact_map.loc[pair], 1) >= 15:
                    sum_top_contacts[num] += contact_map.loc[pair]
                    f.write(name + ',' + pair[0] + '-' + pair[1] + ',' + str(round(contact_map.loc[pair], 1)) + '%\n')
                else:
                    sum_remaining_contacts[num] += contact_map.loc[pair]
            f.write('\n')
    total_contacts = sum(sum_top_contacts) + sum(sum_remaining_contacts)
    f.write('\nType,% of All Contacts\n')
    for name in ['Hydrogen Bonding', 'Hydrophobic', 'Salt Bridges', 'Pi Stacking', 'Pi-Cation']:
        if name not in no_interactions:
            f.write('Hydrogen Bonds,' + str(round(sum_contacts[0] * 100 / total_contacts, 1)) + '%\n')
            f.write('Hydrophobic,' + str(round(sum_contacts[1] * 100 / total_contacts, 1)) + '%\n')
            f.write('Salt Bridges,' + str(round(sum_contacts[2] * 100 / total_contacts, 1)) + '%\n')
            f.write('Pi Stacking,' + str(round(sum_contacts[3] * 100 / total_contacts, 1)) + '%\n')
            f.write('Pi-Cation,' + str(round(sum_contacts[4] * 100 / total_contacts, 1)) + '%\n')
    print('   Saved a report of the top contacts to', os.path.join(saved_results_folder, 'top_contacts.csv'))

print('   ___  _____    ')
print(r' .\'/,-Y"     "~-.  ')
print(r' l.Y             ^.           ')
print(r' /\               _\_  ')
print(r'i            ___/"   "\ ')
print(r'|          /"   "\   o !   ')
print(r'l         ]     o !__./   ')
print(r' \ _  _    \.___./    "~\  ')
print(r'  X \/ \            ___./  ')
print(r' ( \ ___.   _..--~~"   ~`-.  ')
print(r'  ` Z,--   /               \    ')
print(r'    \__.  (   /       ______) ')
print(r'      \   l  /-----~~" /   ')
print(r'       Y   \          / ')
print(r'       |    "x______.^ ')
print(r'       |           \    ')
print('       |            \\')