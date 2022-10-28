# MD-Simulation-Interaction-Analysis

**Analysis script to analyze formation and breakage of non-covalent interactions during MD simulation**

**Automatically plot the following data:**
1. Hydrogen bonds
2. Hydrophobic interactions
3. Pi-pi
4. Cation-pi
5. Salt bridges

Note: Pi-pi and cation-pi are plotted together as pi/cation-pi since there are few of them in general.

**Requirements:**

Python3, VMD, OpenBabel, Plip

**Setup:**
1. Install Miniconda3
2. (Optional) Create a new virtual Conda environment
4. Run:
  ```
  conda install openbabel -c conda-forge
  ```
5. Wait for openbabel to finish installing. Run:
  ```
  pip install --global-option=build_ext --global-option="<INCLUDE_DIR>" --global-option="<LIB_DIR>" plip
  ```
  <INCLUDE_DIR> is "-I/home/**USERNAME**/miniconda3/include/openbabel3" 
  
  <LIB_DIR> is "-L/home/**USERNAME**/miniconda3/lib"
  
**Usage:**
1. Place these scripts in directory containing 1 protein structure file and 1 simulation trajectory file.
2. Run this script (Python 3):
   ```
   python3 simulation_interaction_analysis.py -p PSF, --psf PSF     .psf file containing protein structural information
                                              -d DCD, --dcd DCD     .dcd file containing simulation trajectory (any trajectory format will also work)
                                              -s1 SEG1              First segment/chain/subunit to consider for analysis (follows VMD format)
                                              -s2 SEG2              Second segment/chain/subunit to consider for analysis (follows VMD format)
                                              -s1n SEG1_NAME        Name of first segment/chain/subunit to consider for analysis (customized by user)
                                              -s2n SEG2_NAME        Name of second segment/chain/subunit to consider for analysis (customized by user)
                                              -s1x SEG1_EXCLUDE     Exclude this VMD selection from the first segment/chain/subunit
                                              -s2x SEG2_EXCLUDE     Exclude this VMD selection from the second segment/chain/subunit
                                              -t TIME_TOTAL, --time TIME_TOTAL
                                                                    total simulation time of the full provided trajectory (default in ns)
   ```
   Optional arguments:
   ```
                                              -e END_FRAME          analyze to which frame of the simulation
                                              -s STEP, --step STEP  step used when loading trajectory
                                              --intramolecular      account for intramolecular interactions in analysis
                                              --split SPLIT         split each plot into # of smaller plots covering different time periods
                                              -x X_LABEL            label on x-axis (default = Time (ns))
                                              --skipcommand         if toggled, skip running VMD commands to generate input data
                                              --sortseg2            if toggled, sort interacting residues from seg2 instead of seg1 in ascending order
                                              --labelsize SIZE      label font size (default = 20)
   ```
 3. If desired, tweak interaction detection thresholds (e.g, max distance between hydrogen bond donor and acceptor) in the "Thresholds for detection" section in the code.
   
 **Examples:**

In a folder containing one .psf file and one .dcd file of a 1000 ns long simulation, you want to analyze the interactions between segname PROR (named for B1AR) and segname PROA (named for Ga): _python3 simulation_interaction_analysis.py -t 1000 -s1 "segname PROR" -s2 "segname PROA" -s1n B1AR -s2n Ga_

The same as above, but now you want to exclude certain residues that belong to the first segment from analysis: _python3 simulation_interaction_analysis.py -t 1000 -s1 "segname PROR" -s2 "segname PROA" -s1n B1AR -s2n Ga -s1x "(resid 392 to 402) or (resid 255 to 318)"_

The same as above, but you want to read the simulation trajectory every 2 frames instead of all frames: _python3 simulation_interaction_analysis.py -t 1000 -s1 "segname PROR" -s2 "segname PROA" -s1n B1AR -s2n Ga -s1x "(resid 392 to 402) or (resid 255 to 318) -s 2"_

