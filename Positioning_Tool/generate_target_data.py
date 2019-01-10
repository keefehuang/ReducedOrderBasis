# Add files in the parent folder to the path
import sys
import os
from os.path import splitext
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# Library imports
import numpy as np
import pickle
import sys
import fbpca
import importlib

#Local imports
from mapping import *
from reduced_order import *
from preprocessing import *
from writetoOutput import *
from classes import *


class targetData:
    """Class used to generate tracking node target data for positioning tool "reconstruct.py".
    Data is output as a pkl file. This class can generate a new target data file from scratch using
    a binout with a tracking point definition file or it can update an existing target data file.

    MEMBER VARIABLES
    ----------------
    cListnum : dict
        Contains all supported interface commands. Interface command methods are suffixed by "_command"

    target_data : 3-dim vector np.array with length equal to the number of tracking points
        Contains the final position of each tracking point used for reconstruction. Tracking point coordinates are 
        stored [x-coordinate, y-coordinate, z-coordinate]

    target_id_data : string np.array with length equal to the number of tracking points
        Contains the names/ids of each tracking point. Assumption that tracking point ids in tracking point
        definition file is ordered correctly. Tracking point names MUST be in string format.
    
    MEMBER FUNCTIONS
    ----------------
    list_commands_command           : Interface method. Prints list of available interface commands.
    view_node_command               : Interface method. Used to view target_data associated with input tracking node id. 
    list_node_ids_command           : Interface method. Used to view node ids of all tracking nodes.
    edit_node_command               : Interface method. Used to edit target data of single tracking node. 
    write_command                   : Interface method. Used to write new tracking data file.

    edit_nodes                      : Used to edit multiple tracking nodes.
    get_commands                    : Returns dict of available interface commands
    get_data                        : Returns np.array of target data
    get_ids                         : Returns np.array of node ids
    write                           : Writes new tracking data file
    """
    def __init__(self, full_data_input_file, tracking_nodes_file):
        self.cList = ["List Commands", "View Node", "List Tracking Node IDs", "Edit Tracking Node", "Write Target Data File", "Exit"]
        self.cListnum = {"0" : self.list_commands_command, "1" : self.view_node_command, "2" : self.list_node_ids_command, "3" : self.edit_node_command, "4" : self.write_command, "5" : self.exit}
        full_data_input = Input(full_data_input_file, None, True)
        if tracking_nodes_file is None:
            self.target_data, self.target_id_data, _, _ =\
                full_data_input.extract_simple()
        else:   
            print("Extracting full simulation data")
            _, displacement_data, full_id_data, _ = full_data_input.extract_main()
            print("Extracting tracking point ids and weighting functions")
            tracking_node_data = Input(tracking_nodes_file, None)
            self.target_id_data, tracking_node_list, weights = tracking_node_data.extract_tracking_points_and_weights()
            print("Extracting target position data")            
        
            self.target_data, tracking_ids = append_tracking_point_rows(displacement_data[:,-1].reshape((-1,1)), full_id_data, tracking_node_list)
            self.target_data       = self.target_data[tracking_ids,:]

        self.isEdit = True


    def edit_node_command(self):
        '''
        Function used to edit target_data
        '''
        node_id = input("Please list node id.\n")
        data = input("Please input node data in [x, y, z] format, without the square brackets.\n")
        data = np.array([float(i) for i in data.split(",")])

        node_id = np.where(self.target_id_data == node_id)[0]
        self.target_data.reshape((-1,3))[node_id] = data

    def list_commands_command(self):
        print("Please enter number associated with command.\n")
        for i, command in enumerate(self.cList):
            print("     " + str(i) + ") " + str(command))
        print(" ")

    def view_node_command(self):
        '''
        Function used to display node data of given node id
        '''
        isView = True
        while isView:
            node_id_input = input("Which node to display? Enter 'E' or 'e' to exit\n")        
            node_id = np.where(self.target_id_data ==node_id_input)[0]
            print("Node id : " + str(node_id_input))                
            print(self.target_data.reshape((-1,3))[node_id])
            
            if node_id_input.lower() == "e":
                print("Exiting")
                isView = False

    def list_node_ids_command(self):
        '''
        Function used to display all node ids
        '''
        print("Tracking Node IDs:\n")
        for node_id in self.target_id_data:
            print(node_id)

    def write_command(self):
        outputfile = input("Please insert outputfile name without suffix.\n")
        pkl_data = [self.target_id_data, self.target_data]
        with open(outputfile + "_td.pkl", "wb") as f:
            pickle.dump(pkl_data, f)

    def exit(self):
        self.isEdit = False

    def interface(self):
        self.isEdit = True
        self.list_commands_command()
        while self.isEdit:
            command = input("\nNext Command:\n")
            try:
                self.cListnum[command]()
            except:
                print("Command not understood, please reenter\n")
                self.list_commands_command()


    def edit_nodes(self, tracking_node_dict):
        """
        Function used to edit multiple nodes in target_data
        INPUT:
        tracking_node_dict : dict
            contains modified positions for all tracking nodes
        """
        for key in tracking_node_dict:
            data = np.array(tracking_node_dict[key])
            node_id = np.where(self.target_id_data == node_id)[0]
            self.target_data.reshape((-1,3))[node_id] = data



    def get_commands(self):
        '''
        Function returns dict of interface commands
        OUTPUT:
        cListNum : dict
            contains all interface commands
        '''
        return cListnum

    def get_data(self):
        '''
        Returns tracking node data
        OUTPUT:
        target_data : dict
            contains target data of all tracking nodes
        '''
        return self.target_data

    def get_ids(self):
        '''
        Returns tracking node ids
        OUTPUT:
        target_id_data : dict
            contains node ids of all tracking nodes
        '''
        return self.target_id_data

    def write(self, outputfile):
        '''
        Writes new target data file
        INPUT
        outputfile : string
            name of output file
        '''
        pkl_data = [self.target_id_data, self.target_data]
        with open(outputfile, "wb") as f:
            pickle.dump(pkl_data, f)    

def main():
    description = "Creates/Modifies a target data pkl file"
    epilog = """example:
     # Creation of a new Target Data pkl file
     # $ python generate.py sample.binout tracking_nodes_definition.pkl output_name.pkl
     # $ python generate.py target_data.pkl
     
     notes: - """.format("generate_target_data.py")

    argparser = argparse.ArgumentParser(description=description,epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("inputfile",type=str,metavar="main.binout",help="Input data or existing tracking node positions")
    argparser.add_argument("-t", "--target",dest="trackingnodedef", default=None, metavar="tracking_node_definition.pkl", help="File defining tracking nodes")
    
    args = argparser.parse_args(args=None if len(sys.argv) > 1 else ['--help'])
    
    # Call the generate function.
    target = targetData(args.inputfile, args.trackingnodedef)
    target.interface()
    print("Done")

if __name__ == '__main__':
    main()
