import os, sys
sys.path.append('include')

import h5py
import numpy as np
from copy import deepcopy
import subprocess
from lxml import etree
import re
import math

from   qd.cae.dyna       import Binout
from Binout_reading import binout_reading
from small_func import *
import re



element_keywords = {0 : "*ELEMENT_SHELL", 1 : "*ELEMENT_SOLID", 2 : "*ELEMENT_SEATBELT", 3 : "ELEMENT_BEAM"}


def node_data_to_ticket(node_data):
    '''
    Converts node_data output from get_ndoes() to a ticket
    '''
    ticket = '*NODE\n'
    for i in range(len(node_data[0])):
        ticket += str(node_data[1][i]).rjust(8)
        node_pos = re.findall("[0-9\.\-Ee\+]+", str(node_data[0][i]))
        ticket += node_pos[0].rjust(16) + node_pos[1].rjust(16) + node_pos[2].rjust(16)
        ticket += '       0       0\n'

    return ticket

def element_data_to_ticket(element_data, rm_elems=None):
    '''
    Converts node_data output from get_ndoes() to a ticket
    '''
    if rm_elems is not None:
        element_data = element_data[0][rm_elems,:]
    else:
        element_data = element_data[0]
    eltype = element_data[0][1]
    ticket = element_keywords[eltype] + "\n"

    for element in element_data:
        if eltype != element[1]:
            eltype = element[1]
            ticket += element_keywords[eltype] + "\n"
        ticket += str(element[0]).rjust(8)
        i = 2
        while element[i] != -1 and i < len(element) - 1:
            ticket += str(element[i]).rjust(8)
            i += 1
        ticket += "\n"
        
    return ticket
    
def out_of_range(ext_nid, node_range):
    try:
        return (ext_nid < node_range[0] or ext_nid > node_range[1])
    except:
        return False

def bv2matrix(vec):
   """
   Rearranges an xyz-xyz-xyz basis vector into a matrix:
   n1     x  y  z
   n2     x  y  z
   ..     .  .  .
   nN     x  y  z
   """
   size    = len(vec)
   n_nodes = int(size/3)
   assert n_nodes*3 == size
   
   matrix  = np.empty((n_nodes, 3))
   for node in range(0, n_nodes):
       matrix[node,:] = vec[node*3:node*3+3]
   
   return matrix

def string_slices(inputstring, field_widths, sep=','):
   # Returns slices of the input string, with widths specified. Additional separator can be provided (default:comma).
    pos = 0
    slices = []
    fixed_width = True
    
    if sep in inputstring:
        slices = inputstring.strip().split(sep)
        fixed_width = False
    else:
        for length in field_widths:
            slices.append(inputstring[pos:pos + length].strip())
            pos += length
    
    return slices, fixed_width

class inputdeck:
    """ Class that handles LS-DYNA input deck (*.k or *.key files) modifications.
    Designed for the purpose of changing the reduced set of the ECSW method.
    However, the class can be extended and used for any operations on input
    decks (e.g. reading the geometry of the model directly from the input deck).
    
    MEMBER VARIABLES
    ----------------
    file_content : string
        Content of the input deck file.

    tickets: int list
        Contains list of ints corresponding to all tickets in inputdeck. Each ticket contains the name and contents of a keyword in the inputdeck

    <keyword>pos : int list
        Contains list of ints corresponding to all tickets with a specific keyword, see below for specific keyword

    elementdata: numpy matrix [n_elements x 11]
            ele 1:     EID, ElementType, PID, n1, n2, n3, n4, n5, n6, n7, n8
            ele 2:     EID, ElementType, PID, n1, n2, n3, n4, n5, n6, n7, n8
            
    nodeids:  numpy array (dimension n_nodes, ): contains mapping between internal (indices) and external (values) node IDs.
    
    exteid2int : dictionary, contains a reverse mapping from external to internal element IDs for improved speed
    extnid2int : dictionary, contains a reverse mapping from external to internal node IDs for improved speed
    
    nodedata: numpy matrix [n_nodes x 3], caution: the nodes are ordered in the order of internal IDs (in the order they were read)
                node 1:    X, Y, Z
                node 2:    X, Y, Z
    
    part_contents: dictionary
                 Each key is a regular part ID, the content is the part string
                 
    mat_contents: dictionary
                 Each key is a regular material ID, the content is the material string
    
    shadowparts: dictionary
                            Each key is a regular part ID.
                            The corresponding value is the corresponding null part ID which was created.
    
    shadowmats: dictionary
                            Each key is a regular material ID.
                            The corresponding value is the corresponding null material ID which was created.
                
    shadowoffset: int
                                offset for newly created shadow material and part IDs

    MEMBER FUNCTIONS
    ----------------
    __init__                    : constructor, reads input file
    read_nodes                  : reads and saves nodes of the model
    read_elements               : reads and saves elements of the modes
    reduce_nodes                : creates alternative node tickets with the reduced nodes only
    write_new_content           : writes new modified content to file

    create_xdmf                 : writes an XDMF3 type file and appends a field output for the element weights
    create_timeseries_xdmf      : writes an XDMF3 type file and appends a field output for the element weights, allows for timeseries data to be added
    write_reference_inputdeck   : writes reference inputdeck 
    write_galerkin_inputdeck    : writes Galerkin inputdeck 
    write_reduced_inputdeck     : writes reduced inputdeck, only contains specified nodes/elements
    modify_nodes                : modified internal nodes
    """

    def __init__(self, file, shadowoffset=1000, lastECSWElement=99999999):
        """ Constructor. Reads the input deck file and calls init_tickets().
        Input:  file name of an input deck (string)"""
        self.filename         = file
        self.file_content     = []
        self.file_content_new = []
        self.ecsw_elementtypes= []
        self.lastECSWElement  = lastECSWElement
        
        # ticket positions in self.tickets, list positions of tickets for specific keywords
        self.tickets          = []          # list of all tickets
        self.curveticketpos   = []          # *define_curve
        self.nodeticketpos    = []          # *node
        self.elementticketpos = []          # *element
        self.partticketpos    = []          # *part
        self.matticketpos     = []          # *mat
        self.solidsetticketpos  = []        # *set_solid
        self.eigvalticketpos  = []          # *control_implicit
        self.timestepticketpos= []          # *control_timestep
        self.initialpos       = []          # *initial_velocity
        self.nodesetticketpos = []          # *set_node / *set_node_list
        self.nodecolticketpos = []          # *set_node_column
        self.nodegeneralticketpos = []      # *set_node_general
        self.shellsetticketpos = []         # *set_shell / *set_shell_list
        self.shellcolticketpos = []         # *set_shell_column
        self.shellgeneralticketpos = []     # *set_shell_general
        self.boundaryticketpos = []         # *boundary_spc_node
        self.constrainedlinearpos = []      # *constrained_linear
        self.constrainedshellpos = []       # *constrained_shell
        self.elementmasspos = []            # *element_mass
        self.elementinertiapos = []         # *element_interia
        self.includeticketpos = []          # *include

        self.referenced_nodes = []          
        self.implicit_crvs    = []          # Identify the curve ID for matrix dumps.
        self.curve_ids        = None        # List of curve ids
        self.nodeids          = None        # List of node ids
        self.node_count        = 0          # Total number of nodes in inputdeck
        self.includes         = []          # List of inputdeck objects included in this inputdeck

        self.nodedata         = None        # Node data for all nodes in this inputdeck
        self.elementdata      = None        # Element data for all elements in this inputdeck
        self.part_contents    = {}          
        self.mat_contents     = {}
        self.shadowparts      = {}
        self.shadowmats       = {}
        self.shadowoffset     = shadowoffset   # ID offset for newly created shadow parts and materials
        self.hasRigidWall     = False

        self.timedata         = None

        self.extnid2int       = dict() 
        self.exteid2int       = dict()

        ticket_started = False
        ticket_number  = 0
        # Read input deck
        # TODO: automatically read *SET_ Keywords looking for the contact set, so that it does not need to be passed as an argument from the outside.
        # So far, sets are simply skipped and their content is not modified.
        i = 0
        
        is_include_keyword = False # TODO: extend abilities to work with *INCLUDE keywords.
        with open(file, "r") as f:
            for linecontent in f:
                if linecontent.startswith("*"):
                    lc_line = linecontent.lower()
                else:
                    lc_line = linecontent
                    
                self.file_content.append(lc_line)
                
                if ticket_started and not lc_line.startswith("*"):
                    self.tickets[-1].append(lc_line)
                else:
                    if lc_line.startswith("*"):
                        ticket_started = True
                        self.tickets.append([lc_line])
                        if lc_line.startswith("*node"):
                            self.nodeticketpos.append(ticket_number)
                        elif lc_line.startswith("*element"):
                            if "mass" in lc_line:
                                if "set" not in lc_line:
                                    self.elementmasspos.append(ticket_number)
                            elif "inertia" in lc_line:
                                self.elementinertiapos.append(ticket_number)
                            else:
                                self.elementticketpos.append(ticket_number)
                        elif lc_line.startswith("*part"):
                            self.partticketpos.append(ticket_number)
                        elif lc_line.startswith("*mat"):
                            self.matticketpos.append(ticket_number)
                        elif lc_line.startswith("*set_node"):
                            if "generate" in lc_line:
                                pass
                            elif "general" in lc_line:
                                self.nodegeneralticketpos.append(ticket_number)
                            elif "column" in lc_line:
                                self.nodecolticketpos.append(ticket_number)
                            else:    
                                self.nodesetticketpos.append(ticket_number)
                        elif lc_line.startswith("*set_shell"):
                            if "generate" in lc_line:
                                pass
                            elif "general" in lc_line:
                                self.shellgeneralticketpos.append(ticket_number)
                            elif "column" in lc_line:
                                self.shellcolticketpos.append(ticket_number)
                            else:    
                                self.shellsetticketpos.append(ticket_number)
                        elif lc_line.startswith("*set_solid"):
                            self.solidsetticketpos.append(ticket_number)
                        elif lc_line.startswith("*control_implicit"):
                            self.eigvalticketpos.append(ticket_number)
                        elif lc_line.startswith("*define_curve"):
                            self.curveticketpos.append(ticket_number)
                        elif lc_line.startswith("*control_timestep"):
                            self.timestepticketpos.append(ticket_number)
                        elif lc_line.startswith("*rigidwall_planar"):
                            self.hasRigidWall = True
                        elif lc_line.startswith("*boundary_spc_node"):
                            self.boundaryticketpos.append(ticket_number)
                        elif lc_line.startswith("*constrained_linear"):
                            self.constrainedlinearpos.append(ticket_number)
                        elif lc_line.startswith("*constrained_shell"):
                            self.constrainedshellpos.append(ticket_number)
                        elif lc_line.startswith("*initial_velocity_node"):
                            self.initialpos.append(ticket_number)
                        elif lc_line.startswith("*include"):
                            self.includeticketpos.append(ticket_number)
                            is_include_keyword = True

                            
                        ticket_number += 1
                i += 1
        self.get_all_curve_ids()
        
        # Identify the curve ID for matrix dumps.
        # CAUTION: It will be deleted from the input decks when output_K = False.
        for j in self.eigvalticketpos:
            ticket = self.tickets[j]
            for l, line in enumerate(ticket):
                if line.startswith("*control_implicit_eigenvalue"):
                    k = l+1
                    while True:
                        nextline = ticket[k]
                        if nextline.startswith("$") or nextline.startswith("*"):
                            k+=1
                        else:
                            break
                    CID, _ = string_slices(nextline, [10], sep=',')
                    CID = int(CID[0])
                    if CID < 0:
                        self.implicit_crvs.append(-CID)
        
        # Append the *DEFINE_CURVE keywords used by *CONTROL_IMPLICIT to the eigenvalue tickets.              
        for implicit_crv_id in self.implicit_crvs:  
            # Identify the position of the associated *DEFINE_CURVE keyword.
            if self.curve_ids is not None:
                pos = self.curve_ids.index(implicit_crv_id)
                self.eigvalticketpos.append(self.curveticketpos[pos])
       
        self.read_nodes()
        self.read_elements()
        self.read_includes()

    def read_includes(self):
        include_tickets = [self.tickets[i] for i in self.includeticketpos]
        for ticket in include_tickets:
            self.includes.append(inputdeck(ticket[1].strip()))

    def read_nodes(self):
        """ Read NODE card of the input deck and save Node IDs and coordinates."""
        node_keyword_started = False
        node_count = 0
        
        node_tickets = [self.tickets[i] for i in self.nodeticketpos]

        # Loop over all tickets
        for ticket in node_tickets:
            # First loop to count the nodes
            for i, linecontent in enumerate(ticket):
                if node_keyword_started:
                    if linecontent.startswith("*node"):
                        # already processing previous node keyword, can ignore the second keyword and move straight to the data
                        continue
                    #elif linecontent.startswith("*"):
                    #   # new keyword which is not a node keyword!
                    #   node_keyword_started = False
                    #   continue
                    elif linecontent.startswith("$"):
                        # ignore commented lines
                        pass
                    else:
                        # actual data is here.
                        node_count += 1
                else:
                    # waiting for next *NODE keyword
                    if linecontent.startswith("*node"):
                         node_keyword_started = True
        
        self.nodedata = np.empty((node_count,3), dtype=float)
        self.nodeids  = np.empty(node_count, dtype=int)
        
        # Loop over all tickets
        int_nid = 0
        for ticket in node_tickets:
            # Second loop to save the information in the nodedata array
            node_keyword_started = False
            for i, linecontent in enumerate(ticket):
                if node_keyword_started:
                    if linecontent.startswith("*node"):
                        # already processing previous node keyword, can ignore the second keyword and move straight to the data
                        pass
                    #elif linecontent.startswith("*"):
                    #   # new keyword which is not a node keyword!
                    #   node_keyword_started = False
                    elif linecontent.startswith("$"):
                        # ignore commented lines
                        pass
                    else:
                        # Read and save actual data.
                        slices, _  = string_slices(linecontent, [8, 16, 16, 16], sep=',')
                        # print('index %d, reading slice: %s.' % (int_nid, str(slices)))
                        ext_nid = int(slices[0])
                        x_coor  = float(slices[1])
                        y_coor  = float(slices[2])
                        z_coor  = float(slices[3])
                        self.nodedata[int_nid,:] = np.array([x_coor, y_coor, z_coor])
                        self.nodeids[int_nid]    = ext_nid
                        self.extnid2int[ext_nid] = int_nid
                        
                        int_nid +=1
                else:
                    # waiting for next *NODE keyword
                    if linecontent.startswith("*node"):
                         node_keyword_started = True


    def read_elements(self):
        """ Read ELEMENT cards of the input deck and save element IDs, nodes, and corresponding PIDs"""
        element_keyword_started = False
        element_count = 0
        
        element_tickets = [self.tickets[i] for i in self.elementticketpos]

        # Loop over all tickets
        for ticket in element_tickets:
        # First loop to count the elements
            for i, linecontent in enumerate(ticket):
                if element_keyword_started:
                    if linecontent.startswith("*element_shell") or linecontent.startswith("*element_solid"):
                        # already processing previous element keyword, can ignore the second keyword and move straight to the data
                        pass
                    elif linecontent.startswith("*"):
                        # new keyword which is not a element keyword!
                        element_keyword_started = False
                    elif linecontent.startswith("$"):
                        # ignore commented lines
                        pass
                    else:
                        # actual data is here.
                        element_count += 1
                else:
                    # waiting for next *ELEMENT keyword
                    if linecontent.startswith("*element_shell") or linecontent.startswith("*element_solid"):
                         element_keyword_started = True

        self.elementdata = np.empty((element_count,11), dtype=int)

        # Loop over all tickets
        int_eid = 0
        for ticket in element_tickets:
            # Second loop to save the information in the elementdata array, here we need to distinguish between different element types!
            element_keyword_started = False
            eltype = ""
            for i, linecontent in enumerate(ticket):
                if element_keyword_started:
                    if (eltype is "solid" and linecontent.startswith("*element_solid")) or (eltype is "shell" and linecontent.startswith("*element_shell")):
                        # already processing previous element keyword, can ignore the second keyword and move straight to the data
                        pass
                    elif (eltype is "solid" and linecontent.startswith("*element_shell")):
                        eltype = "shell"
                    elif (eltype is "shell" and linecontent.startswith("*element_solid")):
                        eltype = "solid"
                    elif linecontent.startswith("*"):
                        # new keyword which is not a element keyword!
                        element_keyword_started = False
                    elif linecontent.startswith("$"):
                        # ignore commented lines
                        pass
                    else:
                        # Read and save actual data.
                        if eltype is "shell":        # internal code: shell = 0
                            # 4 nodes
                            slices, _  = string_slices(linecontent, [8, 8, 8, 8, 8, 8], sep=',')
                            
                            isTria = len(slices) <= 5
                            ext_eid = int(slices[0])
                            
                            if isTria:
                                self.elementdata[int_eid,0:11] = np.array([ext_eid, 0, slices[1], slices[2], slices[3], slices[4], slices[4], -1, -1, -1, -1])
                            else:
                                self.elementdata[int_eid,0:11] = np.array([ext_eid, 0, slices[1], slices[2], slices[3], slices[4], slices[5], -1, -1, -1, -1])
                                isTria = len(np.unique(list(map(int, slices[2:6])))) < 4
                            
                            if isTria:
                                if "SHELL3_ELFORM16" not in self.ecsw_elementtypes and ext_eid <= self.lastECSWElement:
                                    self.ecsw_elementtypes.append("SHELL3_ELFORM16")
                            else:
                                if "SHELL4_ELFORM16" not in self.ecsw_elementtypes and ext_eid <= self.lastECSWElement:
                                    self.ecsw_elementtypes.append("SHELL4_ELFORM16")
                            
                        elif eltype is "solid":
                            # 8 nodes                  # internal code: shell = 1
                            slices, _  = string_slices(linecontent, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8], sep=',')
                            ext_eid = int(slices[0])
                            self.elementdata[int_eid,0:11] = np.array([ext_eid, 1, slices[1], slices[2], slices[3], slices[4], slices[5], slices[6], slices[7], slices[8], slices[9]])
                        
                            if "SOLID8_ELFORM2" not in self.ecsw_elementtypes and ext_eid <= self.lastECSWElement:
                                self.ecsw_elementtypes.append("SOLID8_ELFORM2")

                        elif eltype is "seatbelt":
                            # 2 nodes                   #internal code: seatbelt = 2
                            slices, _ = string_slices(linecontent, [8, 8, 8, 8], sep=',')
                            ext_eid = int(slices[0])
                            self.elementdata[int_eid,0:11] = np.array([ext_eid, 2, slices[1], slices[2], slices[3], -1, -1, -1, -1, -1])    
                            
                        
                        elif eltype is "beam":
                            slices, _ = string_slices(linecontent, [8, 8, 8, 8, 8], sep=',')
                            ext_eid = int(slices[0])

                            isDuo = len(slices) < 5
                            if isDuo:
                               self.elementdata[int_eid,0:11] = np.array([ext_eid, 3, slices[1], slices[2], slices[3], -1, -1, -1, -1, -1])    
                            else:
                                self.elementdata[int_eid,0:11] = np.array([ext_eid, 3, slices[1], slices[2], slices[3], slices[4], -1, -1, -1, -1])    

                        self.exteid2int[ext_eid] = int_eid
                        int_eid +=1
                else:
                    # waiting for next *NODE keyword
                    if linecontent.startswith("*element_shell"):
                         element_keyword_started = True
                         eltype = "shell"
                    elif linecontent.startswith("*element_solid"):
                         element_keyword_started = True
                         eltype = "solid"

    def create_shadow_part(self, pid):
        '''
        Copies a part keyword with a given ID and assigns the copy a corresponding NULL material.
        Part tickets are organized as follows:
        *PART
        <CARD 1: TITLE>
        <CARD 2: PID, SECID, MID, EOSID, HGID, GRAV, ADP0PT, TMID>
        
        RETURNS: - new_part_ticket: list of strings which are the lines of the newly created shadow part ticket
                         - new_mat_ticket : list of strings which are the lines of the newly created shadow material ticket
        '''
        part_tickets = [self.tickets[i] for i in self.partticketpos]
        titlepos = None
        
        # Find the correct part
        for ticket in part_tickets:
            # read it, split it
            current_card = 0
            for i, linecontent in enumerate(ticket):
                if (linecontent.startswith("*part")):
                    pass
                elif (linecontent.startswith("$")):
                    pass
                else:
                    current_card += 1
                    
                    if current_card == 1:
                        titlepos = i

                    if current_card == 2:
                        slices, fixed_width = string_slices(linecontent, [10,10,10,10,10,10,10,10], sep=',')
                        part_id = int(slices[0])
                        mid = int(slices[2])
                        
                        if (part_id == pid):
                            # found the correct part. now create a shadow part!
                            new_part_ticket  = deepcopy(ticket)
                            shadow_pid = self.shadowoffset
                            self.shadowparts[pid] = shadow_pid
                            
                            # Create corresponding shadow material.
                            if mid not in self.shadowmats:
                                new_mat_ticket = self.create_shadow_mat(mid)
                                shadow_mid = self.shadowoffset
                            else:
                                new_mat_ticket = []
                                shadow_mid = self.shadowmats[mid]
                                
                            shadow_pid_str = str(shadow_pid)
                            shadow_mid_str = str(shadow_mid)
                            if fixed_width:
                                new_part_ticket[i] = "{:>10}".format(shadow_pid_str) + new_part_ticket[i][10:20] + "{:>10}".format(shadow_mid_str) + new_part_ticket[i][30:]
                            else:
                                slices[0] = shadow_pid_str
                                slices[2] = shadow_mid_str
                                new_part_ticket[i] = ",".join(slices)
                            
                            # Rename the part
                            new_part_ticket[titlepos] = "ECSW_CONTACT_" + new_part_ticket[titlepos]
                            
                            self.shadowoffset += 1
                            return new_part_ticket, new_mat_ticket

    def create_shadow_mat(self, original_mid):
        '''
        Creates a new null material card and attempts to transfer physical properties from the specified material.
        These include density, Young's modulus and Poisson's ratio.
        
        Currently supported materials:
        - MAT_001 (MAT_ELASTIC)
        - MAT_003 (MAT_PLASTIC_KINEMATIC)
        - MAT_020 (MAT_RIGID)
        - MAT_024 (MAT_PIECEWISE_LINEAR_PLASTICITY)

        RETURNS: - new_mat_ticket : list of strings which are the lines of the newly created shadow material ticket
        '''
        
        mat_tickets = [self.tickets[i] for i in self.matticketpos]
        mat_type       = None
        density        = None
        youngs_modulus = None
        poisson_ratio  = None
        
        # find the correct material
        for ticket in mat_tickets:
            # read it, split it
            current_card = 0
            for i, linecontent in enumerate(ticket):
                if (linecontent.startswith("*mat")):
                    mat_type = linecontent.split("*")[-1].rstrip()
                elif (linecontent.startswith("$")):
                    pass
                else:
                    current_card += 1
                    
                    if ("title" in mat_type and current_card == 2) or ("title" not in mat_type and current_card == 1):
                        # Now discern between different material types to read the correct density, young's modulus and poisson's ratio
                        if mat_type == "mat_elastic" or mat_type == "mat_001":
                            slices, _ = string_slices(linecontent, [10,10,10,10], sep=',')
                            mid            = int(slices[0])
                            density        = slices[1]
                            youngs_modulus = slices[2]
                            poisson_ratio  = slices[3]
                        elif mat_type == "mat_plastic_kinematic" or mat_type == "mat_003":
                            slices, _ = string_slices(linecontent, [10,10,10,10], sep=',')
                            mid            = int(slices[0])
                            density        = slices[1]
                            youngs_modulus = slices[2]
                            poisson_ratio  = slices[3]
                        elif mat_type == "mat_rigid" or mat_type == "mat_020":
                            slices, _ = string_slices(linecontent, [10,10,10,10], sep=',')
                            mid            = int(slices[0])
                            density        = slices[1]
                            youngs_modulus = slices[2]
                            poisson_ratio  = slices[3]
                        elif mat_type == "mat_piecewise_linear_plasticity" or mat_type == "mat_024":
                            slices, _ = string_slices(linecontent, [10,10,10,10], sep=',')
                            mid            = int(slices[0])
                            density        = slices[1]
                            youngs_modulus = slices[2]
                            poisson_ratio  = slices[3]
                        else:
                            print("WARNING! Unsupported material " + mat_type + "! Trying my best...")
                            slices, _ = string_slices(linecontent, [10,10,10,10], sep=',')
                            mid            = int(slices[0])
                            density        = slices[1]
                            youngs_modulus = slices[2]
                            poisson_ratio  = slices[3]
                        
                        if mid == original_mid:
                            # Found the target material, start writing corresponding MAT_NULL ticket
                            shadow_mid_str = str(self.shadowoffset)
                            self.shadowmats[mid] = self.shadowoffset
                            
                            mat_null_ticket = []
                            mat_null_ticket.append("*MAT_NULL\n")
                            mat_null_ticket.append("$...>....1....>....2....>....3....>....4....>....5....>....6....>....7....>....8\n")
                            mat_null_ticket.append("$      MID        RO        PC        MU     TEROD     CEROD        YM        PR\n")
                            mat_null_ticket.append("{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(shadow_mid_str, density, "0", "0", "0", "0", youngs_modulus, poisson_ratio))
                            return mat_null_ticket
    
    def ext2int(self, data, ext2int, rm=None, index=None):
        if index is None:
            index = range(len(ext2int[0]))
        if rm is None:
            return data[:,index]
        rm = [ext2int[i] for i in rm]
        data = data[rm, :]
        return data[:,index]
        
    def reduce_nodes(self, rm_element_list, node_range, output_tickets, kept_ext_nodes = []):
        '''
        Reduces the nodes by removing any node which does not belong to an element
        
        INPUTS:
        rm_element_list:         list (array-like), containing the internal IDs of the elements which belong to the reduced mesh.
        node_range:              tuple: (nid_min, nid_max), specifying the range of internal node IDs which are used for the reduction.
        
        OUTPUTS:
        rm_node_ids
        rm_node_ids_ext
        omitted_nodes_int
        omitted_nodes_ext        list of external node IDs which have been removed.
        '''
        new_node_tickets    = [deepcopy(output_tickets[i]) for i in self.nodeticketpos]
        rm_elements         = self.elementdata[rm_element_list,:]
        rm_node_ids         = []
        rm_node_ids_ext     = []
        
        for int_nid, ext_nid in enumerate(self.nodeids):
            if ext_nid in rm_elements[:,3:] or (ext_nid < node_range[0] or ext_nid > node_range[1]) or ext_nid in kept_ext_nodes:
                rm_node_ids.append(int_nid)
                rm_node_ids_ext.append(ext_nid)
        
        rm_node_ids.sort()
        rm_node_ids_ext.sort()
        
        # loop through all node tickets
        omitted_nodes_ext = []
        omitted_nodes_int = []
        int_nid = 0
        for ticket in new_node_tickets:
            try:
                node_keyword_started = False
                for i, linecontent in enumerate(ticket):
                    # print("reading line %d" % i, ", content: " + linecontentecsw)
                    if node_keyword_started:
                        if (linecontent.startswith("*node")):
                            # already processing a node keyword, can ignore the second keyword and move straight to the data
                            pass
                        elif linecontent.startswith("*"):
                            # new keyword which is not a node keyword!
                            node_keyword_started = False
                        elif linecontent.startswith("$"):
                            # ignore commented lines
                            pass
                        else:
                            if int_nid not in rm_node_ids:
                                # Comment out the node
                                ticket[i] = "$ {}".format(ticket[i])
                                omitted_nodes_ext.append(self.nodeids[int_nid])
                                omitted_nodes_int.append(int_nid)
                            int_nid +=1
                    else:
                        # waiting for next *NODE keyword
                        if linecontent.startswith("*node"):
                            node_keyword_started = True
                            
            except Exception as e:
                print("Error occurred in the following keyword, line #%d" %i)
                print("Error: " + str(e))
                print("============= COMPLETE TICKET =============")
                print(ticket)
                print("Exiting inputdeck_writer.py after error.")
                return
        
        omitted_nodes_int.sort()
        return rm_node_ids, rm_node_ids_ext, omitted_nodes_int, omitted_nodes_ext, new_node_tickets

    
    def create_xdmf(self, ECSW_weights=None, Vdict = None, xdmffile=None, pids=None, reduceNodes=False, lastECSWElement=999999999):
        """
        Produces an XDMF3 type file and appends a field output for the element weights.
        The weights can then be visualized in ParaView.
            
        
        INPUTS
        - ECSW_weights:      dictionary of weighting factors. Contains the external element IDs and their corresponding weights
                             {weights description: {element ID (external) : corresponding ECSW weight}}
        - Vdict:             dictionary of reduced bases: {V description: V}, where V is an m x k numpy array
        - xdmffile:          String; path to the output file which will be created
                             If not provided, the file is saved under the same name as d3plot,
                             but with a different ending.
        - pids:              array-like (int); list of part ids which should be used for the visualization.
                             By default, all parts are transferred to the XDMF file.
        - reduceNodes:       specifies whether the nodes not included in the ECSW RM are included in the XDMF output.
        - pids:              array-like (int); controls which PIDs are included in the output.
                             By default, all PIDs are included
        - lastECSWElement:   last ECSW element ID (external).
        
        [NO OUTPUTS]
        """
        
        if ECSW_weights is not None:
            n_different_weights = len(ECSW_weights.keys())
            if Vdict == None and n_different_weights == 0:
                print('Neither a reduced basis nor an ECSW element weight set has been provided.\nSkipping, since cannot produce meaningful output.')
                return
            if n_different_weights > 0:
                write_weights = True
                #has_rot = not isinstance(ECSW_weights[list(ECSW_weights.keys())[0]], float)
        else:
            write_weights = False
            if Vdict == None:
                print('Neither a reduced basis nor an ECSW element weight set has been provided.\nSkipping, since cannot produce meaningful output.')
                return

        if pids is None:
            output_eldata = self.elementdata
        else:
            output_ele_rows = np.unique(np.asarray([i for i in range(0, self.elementdata.shape[0]) if self.elementdata[i,2] in pids]))
            output_eldata   = np.array(self.elementdata[output_ele_rows, :])
            
        # Output file names:
        if xdmffile is None:
            xdmffile = self.filename.rsplit('.d3plot',1)[0] + '.xdmf3'
        h5file       = xdmffile.rsplit('.', 1)[0] + '.h5'
        h5           = h5py.File(h5file, "w")
        xdmfdatafile = os.path.basename(xdmffile)
        h5file       = os.path.basename(h5file)
        
        xdmf = etree.Element("Xdmf")
        domain = etree.SubElement(xdmf,"Domain")
        collection = etree.SubElement(domain,"Grid",
                      attrib={"Name":"FE time series","GridType":"Collection","CollectionType":"Temporal"})
        
        grid = etree.SubElement(collection,"Grid",attrib={"Name":"frame 0"})
        etree.SubElement(grid,"Time",attrib={"Value":"{0:.2e}".format(0.0)})
        
        # Create XDMF node information
        print('Processing XDMF node information')
        h5dset = "/node_coordinates/frame_0"
        
        if reduceNodes:
            int_ele_ids = np.unique(np.asarray([i for i in range(0, output_eldata.shape[0]) if output_eldata[i,0] in ECSW_ele_ids]))
            rm_elements = output_eldata[int_ele_ids,:]
            rm_nodes = []
            rm_nodes_ext = []
            
            for int_nid, ext_nid in enumerate(self.nodeids):
                if ext_nid in rm_elements[:,3:] or (ext_nid < node_range[0] or ext_nid > node_range[1]):
                    rm_nodes.append(int_nid)
                    rm_nodes_ext.append(ext_nid)
            
            numnodes = len(rm_nodes)
            h5.create_dataset(h5dset,data=self.nodedata[np.asarray(rm_nodes, dtype=int).sorted(),:],dtype=float)
        else:
            rm_elements = output_eldata
            numnodes = self.nodedata.shape[0]
            h5.create_dataset(h5dset,data=self.nodedata,dtype=float)
            
        foo = etree.SubElement(grid,"Geometry",attrib={"Origin":"","Type":"XYZ"})
        ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                    "Format":"HDF","Precision":"8"})
        ditem.text = h5file + ":" + h5dset
        
        # Create XDMF5 topology information
        print('Processing XDMF topology information')
        h5topodat   = "/topology/data"
        alltopodata = []
        nelemstot   = rm_elements.shape[0]
        
        print('... total of %d elements' % nelemstot)
        
        # initialize field vectors
        pid_vector      = np.empty((nelemstot, ), dtype = np.int32)
        
        in_ECSW_region  = np.zeros((nelemstot, ), dtype = np.int32)
        
        if write_weights:
            weights     = dict()
            was_removed = dict()
            for wtype in ECSW_weights:
                weights[wtype]     = np.ones((nelemstot, ), dtype = float)
                was_removed[wtype] = np.zeros((nelemstot, ), dtype = np.int32)
            
        # The internal element ID order remains preserved in the XDMF.
        for eid_int in range(0, nelemstot):
            element = rm_elements[eid_int,:]
            eid_ext = element[0]
            eltype  = element[1]
            el_pid  = element[2]
            
            if eltype == 0: # SHELL
                if len(np.unique(element[3:7])) == 3:
                    nodedata = element[3:6]
                else:
                    nodedata = element[3:7]
                
                datalength = len(nodedata)
                
                eltopo    = np.empty((len(nodedata)+1,),  dtype = np.int32)
                eltopo[0] = datalength+1
                for i in range(0,datalength):
                    eltopo[i+1] = self.extnid2int[nodedata[i]]
                    
                alltopodata.append(eltopo)
                
            elif eltype == 1: # SOLID
                nodedata = element[3:11]
                datalength = len(nodedata)
                
                eltopo    = np.empty((len(nodedata)+1,),  dtype = np.int32)
                eltopo[0] = datalength+1
                for i in range(0,datalength):
                    eltopo[i+1] = self.extnid2int[nodedata[i]]
                alltopodata.append(eltopo)
            
            else: # assume 2-noded elements
                nodedata = element[3:5]
                datalength = len(nodedata)
                
                eltopo    = np.empty((len(nodedata)+1,),  dtype = np.int32)
                eltopo[0] = datalength+1
                for i in range(0,datalength):
                    eltopo[i+1] = self.extnid2int[nodedata[i]]
                alltopodata.append(eltopo)
            
            pid_vector[eid_int] = el_pid
            if not write_weights:
                continue
            
            for wtype in ECSW_weights:
                if eid_ext in ECSW_weights[wtype]:
                    in_ECSW_region[eid_int]  = 1
                    weights[wtype][eid_int] = ECSW_weights[wtype][eid_ext]
                else:
                    if eid_ext < lastECSWElement:
                        # Element has not been selected by the ECSW sampling algorithm
                        in_ECSW_region[eid_int]  = 1
                        was_removed[wtype][eid_int] = 1

        alltopodata = np.concatenate(alltopodata)
        ntopodat    = len(alltopodata)
        h5.create_dataset(h5topodat,data=alltopodata,dtype=np.int32)
        foo   = etree.SubElement(grid,"Topology",attrib={"Dimensions":str(nelemstot),"Type":"Mixed"})
        ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(ntopodat),
                                 "Format":"HDF","Precision":"4"})
        ditem.text = h5file + ":" + h5topodat
        
        # Create the PID field
        print('Writing PID field array')
        fname = "PID"
        
        h5propdat = "/fields/{0}".format(fname)
        h5.create_dataset(h5propdat,data=pid_vector,dtype=np.int32,compression="gzip")
        foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":fname,"Type":"Scalar"})
        ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                "Format":"HDF","Precision":"4"})
        ditem.text = h5file + ":" + h5propdat
        
        # Create the ECSW weights field
        if write_weights:
            print('Writing ECSW weights for visualization')
            for wtype in ECSW_weights:
                print('---> ' + wtype)
                h5propdat = "/fields/{0}".format(wtype)
                h5.create_dataset(h5propdat,data=weights[wtype],dtype=float,compression="gzip")
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":wtype,"Type":"Scalar"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(nelemstot),
                            "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5propdat
                
                # Create the field indicating which elements have been removed from the mesh. They get a zero weight.
                print('Writing field array: has the element been removed by the sampling algorithm?')
                fname = 'element_removed_' + wtype
                h5propdat = "/fields/{0}".format(fname)
                h5.create_dataset(h5propdat,data=was_removed[wtype],dtype=np.int32,compression="gzip")
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":fname,"Type":"Scalar"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                        "Format":"HDF","Precision":"4"})
                ditem.text = h5file + ":" + h5propdat
        
        # Create the field indicating which element regions are reduced with ECSW.
        print('Writing field array: is the element part of the ECSW region?')
        fname = "is_in_ECSW_region"
        h5propdat = "/fields/{0}".format(fname)
        h5.create_dataset(h5propdat,data=in_ECSW_region,dtype=np.int32,compression="gzip")
        foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":fname,"Type":"Scalar"})
        ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                "Format":"HDF","Precision":"4"})
        ditem.text = h5file + ":" + h5propdat
            
        # Append the basis vectors
        if Vdict is not None:
            print('Writing basis vectors for visualization')
            # iterate over the different provided bases
            for V_type in Vdict:
                print('---> ' + V_type)
                V = Vdict[V_type]
                
                if len(V.shape) > 1:
                    # V is a matrix. Iterate over its columns
                    for col in range(0, V.shape[1]):
                        bv_name     = V_type + '_' + str(col)
                        basisvector = bv2matrix(V[:,col])
                        
                        # Write the reshaped vector to the file
                        h5propdat = "/fields/{0}".format(bv_name)
                        h5.create_dataset(h5propdat,data=basisvector,dtype=float,compression="gzip")
                        foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":bv_name,"Type":"Vector"})
                        ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} 3".format(basisvector.shape[0]),
                                                "Format":"HDF","Precision":"8"})
                        ditem.text = h5file + ":" + h5propdat
                else:
                    # V is a single column vector
                    basisvector = bv2matrix(V)
                    bv_name     = V_type + '_0'
                    
                    # Write the reshaped vector to the file
                    h5propdat = "/fields/{0}".format(bv_name)
                    h5.create_dataset(h5propdat,data=basisvector,dtype=float,compression="gzip")
                    foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":bv_name,"Type":"Vector"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} 3".format(basisvector.shape[0]),
                                            "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5propdat
        
        # Finish the output
        h5.close()
        print("--> {0} {1}".format(xdmffile,h5file))
        with open(xdmffile,"w") as outfile:
            outfile.write( '<?xml version="1.0" encoding="utf-8"?>\n' )
            outfile.write( '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n' )
            outfile.write( etree.tostring(xdmf,pretty_print=True).decode("utf-8") )
    
    def create_timeseries_xdmf(self, model_binout, reference_binout=None, ECSW_weights=None, xdmffile=None,   \
                               pids=None, reduceNodes=False, lastECSWElement=999999999, numnodes=None,        \
                               interp_time = True, hasRot=True, memory_mapped=False):
        """
        Produces an XDMF3 type file and appends a field output for the element weights.
        The weights can then be visualized in ParaView.
            
        INPUTS
        
        - model_binout:      binout file of the model (usually the ROM/HROM which should be reconstructed)
                             It needs to have all nodes (i.e. not an ECSW RN simulation!)
                             
        - reference_binout:  binout file of a reference simulation. If available, its data will also be written to the xdmf
                             along with the distance vectors.
                             ALERT! The reference binout needs to have the same number of time steps.
                             
        - ECSW_weights:      dictionary of weighting factors. Contains the external element IDs and their corresponding weights
                             {weights description: {element ID (external) : corresponding ECSW weight}}
                             
        - xdmffile:          String; path to the output file which will be created
                             If not provided, the file is saved under the same name as d3plot,
                             but with a different ending.
                                     
        - pids:              array-like (int); list of part ids which should be used for the visualization.
                             By default, all parts are transferred to the XDMF file.
                             
        - reduceNodes:       specifies whether the nodes not included in the ECSW RM are included in the XDMF output.
                                     
        - lastECSWElement:   last ECSW element ID (external).
        
        - numnodes:          specifies the last node for the output. Not used at the moment.
        
        - interp_time:       If the output times of the reference and the model binout do not match, linear interpolation is performed.
                             In case interp_time is False, only the "closest" state is used.
        
        - hasRot:            specifies whether or not to include information on nodal rotations in the XDMF output
                             Will crash if set to True and the model does not have any nodal rotations.
        
        - memory_mapped:     Specifies whether or not the function memory-maps the snapshot matrices for reduced RAM consumption.
                             [TODO] Not implemented yet.
        """
        
        if ECSW_weights is not None:
            n_different_weights = len(ECSW_weights.keys())
            if n_different_weights > 0:
                write_weights = True
        else:
            write_weights = False
    
        if pids is None:
            output_eldata = self.elementdata
        else:
            output_ele_rows = np.unique(np.asarray([i for i in range(0, self.elementdata.shape[0]) if self.elementdata[i,2] in pids]))
            output_eldata   = np.array(self.elementdata[output_ele_rows, :])
            
        # Output file names:
        if xdmffile is None:
            xdmffile = self.filename.rsplit('.d3plot',1)[0] + '_timeseries.xdmf3'
        
        h5file         = xdmffile.rsplit('.', 1)[0] + '.h5'
        h5             = h5py.File(h5file, "w")
        reference_xdmf = xdmffile.replace(".xdmf", "_reference.xdmf")
        xdmfdatafile   = os.path.basename(xdmffile)
        h5file         = os.path.basename(h5file)
        
        # Read the time data
        timesteps_model = binout_reading(model_binout, False, "time")
        if reference_binout is not None:
            timesteps_reference = binout_reading(reference_binout, False, "time")
        
        n_iter = 0
        while np.abs(timesteps_reference[-1] - timesteps_model[-1]) / timesteps_model[-1] > 0.05:
            print("Wrong time scales. Attempting to convert.")
            if timesteps_reference[-1] > timesteps_model[-1]:
                timesteps_reference = timesteps_reference / 10
            else:
                timesteps_reference = timesteps_reference * 10
            n_iter +=1
            
            if n_iter > 10:
                print("ERROR! COULD NOT MATCH TIME DATA BETWEEN MODEL AND REFERENCE BINOUT")
                return
        
        xdmf = etree.Element("Xdmf")
        domain = etree.SubElement(xdmf,"Domain")
        collection = etree.SubElement(domain,"Grid",
                      attrib={"Name":"FE time series","GridType":"Collection","CollectionType":"Temporal"})
        
        ref_xdmf   = etree.Element("Xdmf")
        ref_domain = etree.SubElement(ref_xdmf,"Domain")
        ref_collection = etree.SubElement(ref_domain,"Grid",
                         attrib={"Name":"FE time series","GridType":"Collection","CollectionType":"Temporal"})
        
        # Create XDMF node information
        print('Processing XDMF node information')
        if reduceNodes:
            int_ele_ids = np.unique(np.asarray([i for i in range(0, output_eldata.shape[0]) if output_eldata[i,0] in ECSW_ele_ids]))
            rm_elements = output_eldata[int_ele_ids,:]
            rm_nodes = []
            
            for int_nid, ext_nid in enumerate(self.nodeids):
                if ext_nid in rm_elements[:,3:] or (ext_nid < node_range[0] or ext_nid > node_range[1]):
                    rm_nodes.append(int_nid)
            
            rm_nodes = np.sort(np.asarray(rm_nodes, dtype=int))
            numnodes = len(rm_nodes)
            # h5.create_dataset(h5dset,data=self.nodedata[np.asarray(rm_nodes, dtype=int).sorted(),:],dtype=float)
        else:
            rm_elements = output_eldata
            numnodes    = self.nodedata.shape[0]
            rm_nodes    = np.arange(numnodes)
            # h5.create_dataset(h5dset,data=self.nodedata,dtype=float)
                   
        # Create XDMF5 topology information
        print('Processing XDMF topology information')
        h5topodat   = "/topology/data"
        alltopodata = []
        nelemstot   = rm_elements.shape[0]
        
        print('... total of %d elements' % nelemstot)
        
        # initialize field vectors
        pid_vector      = np.empty((nelemstot, ), dtype = np.int32)
        in_ECSW_region  = np.zeros((nelemstot, ), dtype = np.int32)
        
        if write_weights:
            weights     = dict()
            was_removed = dict()
            for wtype in ECSW_weights:
                weights[wtype]     = np.ones((nelemstot, ), dtype = float)
                was_removed[wtype] = np.zeros((nelemstot, ), dtype = np.int32)
            
        # The internal element ID order remains preserved in the XDMF.
        for eid_int in range(0, nelemstot):
            element = rm_elements[eid_int,:]
            eid_ext = element[0]
            eltype  = element[1]
            el_pid  = element[2]
            
            if eltype == 0: # SHELL
                if len(np.unique(element[3:7])) == 3:
                    nodedata = element[3:6]
                else:
                    nodedata = element[3:7]
                
                datalength = len(nodedata)
                
                eltopo    = np.empty((len(nodedata)+1,),  dtype = np.int32)
                eltopo[0] = datalength+1
                for i in range(0,datalength):
                    eltopo[i+1] = self.extnid2int[nodedata[i]]
                    
                alltopodata.append(eltopo)
                
            elif eltype == 1: # SOLID
                nodedata = element[3:11]
                datalength = len(nodedata)
                
                eltopo    = np.empty((len(nodedata)+1,),  dtype = np.int32)
                eltopo[0] = datalength+1
                for i in range(0,datalength):
                    eltopo[i+1] = self.extnid2int[nodedata[i]]
                alltopodata.append(eltopo)
            
            else: # assume 2-noded elements
                nodedata = element[3:5]
                datalength = len(nodedata)
                
                eltopo    = np.empty((len(nodedata)+1,),  dtype = np.int32)
                eltopo[0] = datalength+1
                for i in range(0,datalength):
                    eltopo[i+1] = self.extnid2int[nodedata[i]]
                alltopodata.append(eltopo)
            
            pid_vector[eid_int] = el_pid
            if not write_weights:
                continue
            
            for wtype in ECSW_weights:
                if eid_ext in ECSW_weights[wtype]:
                    in_ECSW_region[eid_int]  = 1
                    weights[wtype][eid_int] = ECSW_weights[wtype][eid_ext]
                else:
                    if eid_ext < lastECSWElement:
                        # Element has not been selected by the ECSW sampling algorithm
                        in_ECSW_region[eid_int]  = 1
                        was_removed[wtype][eid_int] = 1
        
        alltopodata = np.concatenate(alltopodata)
        ntopodat    = len(alltopodata)
        h5.create_dataset(h5topodat,data=alltopodata,dtype=np.int32)
        
        # PID field
        h5pid_name = "PID"
        h5pid_dat  = "/fields/{0}".format(h5pid_name)
        h5.create_dataset(h5pid_dat,data=pid_vector,dtype=np.int32,compression="gzip")
        
        # ECSW weights
        if write_weights:
            print('Writing ECSW weights for visualization')
            for wtype in ECSW_weights:
                print('---> ' + wtype)
                h5propdat = "/fields/{0}".format(wtype)
                h5.create_dataset(h5propdat,data=weights[wtype],dtype=float,compression="gzip")
                
                # Create the field indicating which elements have been removed from the mesh. They get a zero weight.
                print('Writing field array: has the element been removed by the sampling algorithm?')
                fname = 'element_removed_' + wtype
                h5propdat = "/fields/{0}".format(fname)
                h5.create_dataset(h5propdat,data=was_removed[wtype],dtype=np.int32,compression="gzip")
        
        # Create the field indicating which element regions are reduced with ECSW.
        print('Writing field array: is the element part of the ECSW region?')
        fname = "in_ECSW_region"
        h5_is_ecsw_region = "/fields/{0}".format(fname)
        h5.create_dataset(h5_is_ecsw_region,data=in_ECSW_region,dtype=np.int32,compression="gzip")
        
        grids    = []
        refgrids = []
        for timestep, t in enumerate(timesteps_model):
            grid     = etree.SubElement(collection,"Grid",attrib={"Name": "frame %d" % timestep})
            ref_grid = etree.SubElement(ref_collection,"Grid",attrib={"Name": "frame %d" % timestep})
            
            etree.SubElement(grid,"Time",attrib={"Value":"{0:.2e}".format(t)})
            etree.SubElement(ref_grid,"Time",attrib={"Value":"{0:.2e}".format(t)})
                
            # Write topology
            foo   = etree.SubElement(grid,"Topology",attrib={"Dimensions":str(nelemstot),"Type":"Mixed"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(ntopodat),
                                     "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5topodat
            
            foo   = etree.SubElement(ref_grid,"Topology",attrib={"Dimensions":str(nelemstot),"Type":"Mixed"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(ntopodat),
                                     "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5topodat
            
            # PID information
            foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":h5pid_name,"Type":"Scalar"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                    "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5pid_dat
            
            foo = etree.SubElement(ref_grid,"Attribute",attrib={"Center":"Cell","Name":h5pid_name,"Type":"Scalar"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                    "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5pid_dat
            
            # ECSW weights field
            if write_weights:
                for wtype in ECSW_weights:
                    weight_dat = "/fields/{0}".format(wtype)
                    
                    foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":wtype,"Type":"Scalar"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(nelemstot),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + weight_dat
                    
                    foo = etree.SubElement(ref_grid,"Attribute",attrib={"Center":"Cell","Name":wtype,"Type":"Scalar"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(nelemstot),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + weight_dat
                    
                    # Create the field indicating which elements have been removed from the mesh. They get a zero weight.
                    fname = 'element_removed_' + wtype
                    h5propdat = "/fields/{0}".format(fname)
                    foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":fname,"Type":"Scalar"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                             "Format":"HDF","Precision":"4"})
                    ditem.text = h5file + ":" + h5propdat
                    
                    foo = etree.SubElement(ref_grid,"Attribute",attrib={"Center":"Cell","Name":fname,"Type":"Scalar"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                             "Format":"HDF","Precision":"4"})
                    ditem.text = h5file + ":" + h5propdat
                    
            # Field specifying whether elements are in ECSW region
            foo   = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":fname,"Type":"Scalar"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                     "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5_is_ecsw_region
            
            foo   = etree.SubElement(ref_grid,"Attribute",attrib={"Center":"Cell","Name":fname,"Type":"Scalar"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                     "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5_is_ecsw_region
            
            grids.append(grid)
            refgrids.append(ref_grid)
        
        # Displacements and rotations
        model_displacements   = rearange_xyz(binout_reading(model_binout, False, "displacements"))[0:3*numnodes,:]
        if hasRot:
            model_rotations   = rearange_xyz(binout_reading(model_binout, False, "rotations"))[0:3*numnodes,:]
        
        if reference_binout is not None:
            ref_displacements = rearange_xyz(binout_reading(reference_binout, False, "displacements"))[0:3*numnodes,:]
            if hasRot:
                ref_rotations = rearange_xyz(binout_reading(reference_binout, False, "rotations"))[0:3*numnodes,:]
        
        for timestep, t in enumerate(timesteps_model):
            grid    = grids[timestep]
            refgrid = refgrids[timestep]
            
            mod_disp      = bv2matrix(model_displacements[:,timestep])
            # Get new nodal coordinates and save them as h5dset.
            new_coord     = self.nodedata + mod_disp
            if hasRot:
                new_rot   = bv2matrix(model_rotations[:,timestep])
                  
            # Write nodal coordinates for this frame
            h5dset = "/node_coordinates/frame_{}".format(timestep)
            h5.create_dataset(h5dset,data=new_coord[rm_nodes,:],dtype=float)
            foo = etree.SubElement(grid,"Geometry",attrib={"Origin":"","Type":"XYZ"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                     "Format":"HDF","Precision":"8"})
            ditem.text = h5file + ":" + h5dset
            
            # Write nodal displacements
            h5dset = "/node_displacements/frame_{}".format(timestep)
            h5.create_dataset(h5dset,data=mod_disp[rm_nodes,:],dtype=float)
            fname = "displacements"
            foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                     "Format":"HDF","Precision":"8"})
            ditem.text = h5file + ":" + h5dset
            
            if hasRot:
                # Write nodal rotations
                h5dset = "/node_rotations/frame_{}".format(timestep)
                h5.create_dataset(h5dset,data=new_rot[rm_nodes,:],dtype=float)
                fname = "rotations"
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
            
            # Same for reference binout, if available.
            if reference_binout is not None:
                if (timesteps_reference[timestep] - t) / max(t, 1e-16) > 1e-3:
                    # interpolate reference coordinates from two output time steps.
                    
                    if timesteps_reference[timestep] > t and timestep > 0:
                        print('HI, tleft = %2f, t = %2f, tright = %2f' % (timesteps_reference[timestep-1], t, timesteps_reference[timestep]))
                        ref_disp_1 = bv2matrix(ref_displacements[:,timestep-1])
                        ref_disp_2 = bv2matrix(ref_displacements[:,timestep])
                        dt_left    = (t - timesteps_reference[timestep-1]) / (timesteps_reference[timestep] - timesteps_reference[timestep-1])
                    else:
                        print('HI, tleft = %2f, t = %2f, tright = %2f' % (timesteps_reference[timestep], t, timesteps_reference[timestep+1]))
                        ref_disp_1 = bv2matrix(ref_displacements[:,timestep])
                        ref_disp_2 = bv2matrix(ref_displacements[:,timestep+1])
                        dt_left    = (t - timesteps_reference[timestep])   / (timesteps_reference[timestep+1] - timesteps_reference[timestep])
                    
                    ref_disp = (1-dt_left) * ref_disp_1 + dt_left * ref_disp_2
                    if hasRot:
                        if timesteps_reference[timestep] > t and timestep > 0:
                            ref_rot_1 = bv2matrix(ref_rotations[:,timestep-1])
                            ref_rot_2 = bv2matrix(ref_rotations[:,timestep])
                            dt_left    = (t - timesteps_reference[timestep-1]) / (timesteps_reference[timestep] - timesteps_reference[timestep-1])
                        else:
                            ref_rot_1 = bv2matrix(ref_rotations[:,timestep])
                            ref_rot_2 = bv2matrix(ref_rotations[:,timestep+1])
                            dt_left    = (t - timesteps_reference[timestep])   / (timesteps_reference[timestep+1] - timesteps_reference[timestep])
                            
                        ref_rot = (1-dt_left) * ref_rot_1 + dt_left * ref_rot_2
                else:
                    ref_disp = bv2matrix(ref_displacements[:,timestep])
                    if hasRot:
                        ref_rot = bv2matrix(ref_rotations[:,timestep])
                
                ref_coord = self.nodedata + ref_disp
                
                # Reference model coordinates
                h5dset = "/reference_node_coordinates/frame_{}".format(timestep)
                h5.create_dataset(h5dset,data=ref_coord[rm_nodes,:],dtype=float)
                foo = etree.SubElement(refgrid,"Geometry",attrib={"Origin":"","Type":"XYZ"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
            
                # Reference nodal displacements
                h5dset = "/reference_node_displacements/frame_{}".format(timestep)
                h5.create_dataset(h5dset,data=ref_disp[rm_nodes,:],dtype=float)
                fname = "Reference displacements"
                foo = etree.SubElement(refgrid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                # Write the difference vector to reference displacements for this frame
                disp_diff = mod_disp[rm_nodes,:] - ref_disp[rm_nodes,:]
                
                h5dset = "/diff_displacements/frame_{}".format(timestep)
                h5.create_dataset(h5dset, data = disp_diff, dtype=float)
                fname = "Diff displacements"
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                # Write relative error to reference displacements of this frame
                rel_errors = np.divide(np.linalg.norm(disp_diff, axis = 1), np.maximum(np.linalg.norm(ref_disp[rm_nodes,:], axis = 1), 1e-16 * np.ones((disp_diff.shape[0],))))
                h5dset     = "/reldiff_displacements/frame_{}".format(timestep)
                h5.create_dataset(h5dset, data = rel_errors, dtype=float)
                fname = "Relative displacement error to reference"
                foo   = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Scalar"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(numnodes),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                if hasRot:
                    # Write nodal rotations
                    h5dset = "/reference_node_rotations/frame_{}".format(timestep)
                    h5.create_dataset(h5dset,data=ref_rot[rm_nodes,:],dtype=float)
                    fname = "Reference rotations"
                    foo = etree.SubElement(refgrid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
          
                    # Write magnitude of the difference vector to reference rotations for this frame
                    rot_diff = new_rot[rm_nodes,:] - ref_rot[rm_nodes,:]
                    
                    h5dset = "/diff_rotations/frame_{}".format(timestep)
                    h5.create_dataset(h5dset, data = rot_diff, dtype=float)
                    fname = "Diff rotations"
                    foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
                    
                    # Write relative error to reference rotations of this frame
                    rel_errors = np.divide(np.linalg.norm(rot_diff, axis = 1), np.maximum(np.linalg.norm(ref_rot[rm_nodes,:], axis = 1), 1e-16 * np.ones((rot_diff.shape[0],))))
                    h5dset     = "/reldiff_rotations/frame_{}".format(timestep)
                    h5.create_dataset(h5dset, data = rel_errors, dtype=float)
                    fname = "Relative rotation error to reference"
                    foo   = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Scalar"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(numnodes),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
            
        del model_displacements, mod_disp, new_coord
        if hasRot:
            del model_rotations, new_rot
            
        if reference_binout is not None:
            del ref_displacements, ref_disp
            if hasRot:
                del ref_rotations, ref_rot
        
        # Velocities and angular velocities 
        model_vel        = rearange_xyz(binout_reading(model_binout, False, "velocities"))[0:3*numnodes,:]
        if hasRot:
            model_rvel   = rearange_xyz(binout_reading(model_binout, False, "angular_velocities"))[0:3*numnodes,:]
        
        if reference_binout is not None:
            ref_vel      = rearange_xyz(binout_reading(reference_binout, False, "velocities"))[0:3*numnodes,:]
            if hasRot:
                ref_rvel = rearange_xyz(binout_reading(reference_binout, False, "angular_velocities"))[0:3*numnodes,:]
        
        for timestep, t in enumerate(timesteps_model):
            grid    = grids[timestep]
            refgrid = refgrids[timestep]
            
            mod_vdisp      = bv2matrix(model_vel[:,timestep])
            if hasRot:
                mod_vrot   = bv2matrix(model_rvel[:,timestep])
                              
            # Write nodal velocities
            h5dset = "/node_velocities/frame_{}".format(timestep)
            h5.create_dataset(h5dset,data=mod_vdisp[rm_nodes,:],dtype=float)
            fname = "velocities"
            foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                     "Format":"HDF","Precision":"8"})
            ditem.text = h5file + ":" + h5dset
            
            if hasRot:
                # Write nodal rotations
                h5dset = "/node_rot_velocities/frame_{}".format(timestep)
                h5.create_dataset(h5dset,data=mod_vrot[rm_nodes,:],dtype=float)
                fname = "angular velocities"
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
            
            # Same for reference binout, if available.
            if reference_binout is not None:
                if (timesteps_reference[timestep] - t) / max(t, 1e-16) > 1e-3:
                    # interpolate reference coordinates from two output time steps.
                    
                    if timesteps_reference[timestep] > t and timestep > 0:
                        print('HI, tleft = %2f, t = %2f, tright = %2f' % (timesteps_reference[timestep-1], t, timesteps_reference[timestep]))
                        ref_vdisp_1 = bv2matrix(ref_vel[:,timestep-1])
                        ref_vdisp_2 = bv2matrix(ref_vel[:,timestep])
                        dt_left    = (t - timesteps_reference[timestep-1]) / (timesteps_reference[timestep] - timesteps_reference[timestep-1])
                    else:
                        print('HI, tleft = %2f, t = %2f, tright = %2f' % (timesteps_reference[timestep], t, timesteps_reference[timestep+1]))
                        ref_vdisp_1 = bv2matrix(ref_vel[:,timestep])
                        ref_vdisp_2 = bv2matrix(ref_vel[:,timestep+1])
                        dt_left    = (t - timesteps_reference[timestep])   / (timesteps_reference[timestep+1] - timesteps_reference[timestep])
                    
                    ref_vdisp = (1-dt_left) * ref_vdisp_1 + dt_left * ref_vdisp_2

                    if hasRot:
                        if timesteps_reference[timestep] > t and timestep > 0:
                            ref_vrot_1 = bv2matrix(ref_rvel[:,timestep-1])
                            ref_vrot_2 = bv2matrix(ref_rvel[:,timestep])
                            dt_left    = (t - timesteps_reference[timestep-1]) / (timesteps_reference[timestep] - timesteps_reference[timestep-1])
                        else:
                            ref_vrot_1 = bv2matrix(ref_rvel[:,timestep])
                            ref_vrot_2 = bv2matrix(ref_rvel[:,timestep+1])
                            dt_left    = (t - timesteps_reference[timestep])   / (timesteps_reference[timestep+1] - timesteps_reference[timestep])
                            
                        ref_vrot = (1-dt_left) * ref_vrot_1 + dt_left * ref_vrot_2
                else:
                    ref_vdisp = bv2matrix(ref_vel[:,timestep])
                    if hasRot:
                        ref_vrot = bv2matrix(ref_rvel[:,timestep])
                               
                # Reference velocities
                h5dset = "/reference_node_velocities/frame_{}".format(timestep)
                h5.create_dataset(h5dset,data=ref_vdisp[rm_nodes,:],dtype=float)
                fname = "Reference velocities"
                foo = etree.SubElement(refgrid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                # Write the difference vector to reference velocities for this frame
                vdisp_diff = mod_vdisp[rm_nodes,:] - ref_vdisp[rm_nodes,:]
                
                h5dset = "/diff_velocities/frame_{}".format(timestep)
                h5.create_dataset(h5dset, data = vdisp_diff, dtype=float)
                fname = "Diff velocities"
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                # Write relative error to reference velocities of this frame
                rel_errors = np.divide(np.linalg.norm(vdisp_diff, axis = 1), np.maximum(np.linalg.norm(ref_vdisp[rm_nodes,:], axis = 1), 1e-16 * np.ones((vdisp_diff.shape[0],))))
                h5dset     = "/reldiff_velocities/frame_{}".format(timestep)
                h5.create_dataset(h5dset, data = rel_errors, dtype=float)
                fname = "Relative velocity error to reference"
                foo   = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Scalar"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(numnodes),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                if hasRot:
                    # Write angular velocities
                    h5dset = "/reference_node_angular_velocities/frame_{}".format(timestep)
                    h5.create_dataset(h5dset,data=ref_vrot[rm_nodes,:],dtype=float)
                    fname = "Reference angular velocities"
                    foo = etree.SubElement(refgrid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
          
                    # Write magnitude of the difference vector to reference angular velocities for this frame
                    vrot_diff = mod_vrot[rm_nodes,:] - ref_vrot[rm_nodes,:]
                    
                    h5dset = "/diff_angular_velocities/frame_{}".format(timestep)
                    h5.create_dataset(h5dset, data = vrot_diff, dtype=float)
                    fname = "Diff angular velocities"
                    foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
                    
                    # Write relative error to reference angular velocities of this frame
                    rel_errors = np.divide(np.linalg.norm(vrot_diff, axis = 1), np.maximum(np.linalg.norm(ref_vrot[rm_nodes,:], axis = 1), 1e-16 * np.ones((vrot_diff.shape[0],))))
                    h5dset     = "/reldiff_angular_velocities/frame_{}".format(timestep)
                    h5.create_dataset(h5dset, data = rel_errors, dtype=float)
                    fname = "Relative angular velocity error to reference"
                    foo   = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Scalar"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(numnodes),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
        
        # Accelerations and angular accelerations
        del model_vel, mod_vdisp
        if hasRot:
            del model_rvel, mod_vrot
            
        if reference_binout is not None:
            del ref_vel, ref_vdisp
            if hasRot:
                del ref_rvel, ref_vrot
        
        model_acc        = rearange_xyz(binout_reading(model_binout, False, "accelerations"))[0:3*numnodes,:]
        if hasRot:
            model_racc   = rearange_xyz(binout_reading(model_binout, False, "angular_accelerations"))[0:3*numnodes,:]
        
        if reference_binout is not None:
            ref_acc      = rearange_xyz(binout_reading(reference_binout, False, "accelerations"))[0:3*numnodes,:]
            if hasRot:
                ref_racc = rearange_xyz(binout_reading(reference_binout, False, "angular_accelerations"))[0:3*numnodes,:]
        
        for timestep, t in enumerate(timesteps_model):
            grid = grids[timestep]
            refgrid = refgrids[timestep]

            mod_adisp      = bv2matrix(model_acc[:,timestep])
            if hasRot:
                mod_arot   = bv2matrix(model_racc[:,timestep])
                              
            # Write nodal velocities
            h5dset = "/node_accelerations/frame_{}".format(timestep)
            h5.create_dataset(h5dset,data=mod_adisp[rm_nodes,:],dtype=float)
            fname = "accelerations"
            foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
            ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                     "Format":"HDF","Precision":"8"})
            ditem.text = h5file + ":" + h5dset
            
            if hasRot:
                # Write nodal rotations
                h5dset = "/node_rot_accelerations/frame_{}".format(timestep)
                h5.create_dataset(h5dset,data=mod_arot[rm_nodes,:],dtype=float)
                fname = "angular accelerations"
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
            
            # Same for reference binout, if available.
            if reference_binout is not None:
                if (timesteps_reference[timestep] - t) / max(t, 1e-16) > 1e-3:
                    # interpolate reference coordinates from two output time steps.
                    
                    if timesteps_reference[timestep] > t and timestep > 0:
                        print('HI, tleft = %2f, t = %2f, tright = %2f' % (timesteps_reference[timestep-1], t, timesteps_reference[timestep]))
                        ref_adisp_1 = bv2matrix(ref_acc[:,timestep-1])
                        ref_adisp_2 = bv2matrix(ref_acc[:,timestep])
                        dt_left    = (t - timesteps_reference[timestep-1]) / (timesteps_reference[timestep] - timesteps_reference[timestep-1])
                    else:
                        print('HI, tleft = %2f, t = %2f, tright = %2f' % (timesteps_reference[timestep], t, timesteps_reference[timestep+1]))
                        ref_adisp_1 = bv2matrix(ref_ccc[:,timestep])
                        ref_adisp_2 = bv2matrix(ref_ccc[:,timestep+1])
                        dt_left    = (t - timesteps_reference[timestep])   / (timesteps_reference[timestep+1] - timesteps_reference[timestep])
                    
                    ref_adisp = (1-dt_left) * ref_adisp_1 + dt_left * ref_adisp_2

                    if hasRot:
                        if timesteps_reference[timestep] > t and timestep > 0:
                            ref_arot_1 = bv2matrix(ref_racc[:,timestep-1])
                            ref_arot_2 = bv2matrix(ref_racc[:,timestep])
                            dt_left    = (t - timesteps_reference[timestep-1]) / (timesteps_reference[timestep] - timesteps_reference[timestep-1])
                        else:
                            ref_arot_1 = bv2matrix(ref_raccc[:,timestep])
                            ref_arot_2 = bv2matrix(ref_raccc[:,timestep+1])
                            dt_left    = (t - timesteps_reference[timestep])   / (timesteps_reference[timestep+1] - timesteps_reference[timestep])
                            
                        ref_arot = (1-dt_left) * ref_arot_1 + dt_left * ref_arot_2
                else:
                    ref_adisp = bv2matrix(ref_acc[:,timestep])
                    if hasRot:
                        ref_arot = bv2matrix(ref_racc[:,timestep])
                               
                # Reference accelerations
                h5dset = "/reference_node_accelerations/frame_{}".format(timestep)
                h5.create_dataset(h5dset,data=ref_adisp[rm_nodes,:],dtype=float)
                fname = "Reference accelerations"
                foo = etree.SubElement(refgrid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                # Write the difference vector to reference accelerations for this frame
                adisp_diff = mod_adisp[rm_nodes,:] - ref_adisp[rm_nodes,:]
                
                h5dset = "/diff_accelerations/frame_{}".format(timestep)
                h5.create_dataset(h5dset, data = adisp_diff, dtype=float)
                fname = "Diff accelerations"
                foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                # Write relative error to reference accelerations of this frame
                rel_errors = np.divide(np.linalg.norm(adisp_diff, axis = 1), np.maximum(np.linalg.norm(ref_adisp[rm_nodes,:], axis = 1), 1e-16 * np.ones((adisp_diff.shape[0],))))
                h5dset     = "/reldiff_accelerations/frame_{}".format(timestep)
                h5.create_dataset(h5dset, data = rel_errors, dtype=float)
                fname = "Relative acceleration error to reference"
                foo   = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Scalar"})
                ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(numnodes),
                                         "Format":"HDF","Precision":"8"})
                ditem.text = h5file + ":" + h5dset
                
                if hasRot:
                    # Write angular accelerations
                    h5dset = "/reference_node_angular_accelerations/frame_{}".format(timestep)
                    h5.create_dataset(h5dset,data=ref_arot[rm_nodes,:],dtype=float)
                    fname = "Reference angular accelerations"
                    foo = etree.SubElement(refgrid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
          
                    # Write magnitude of the difference vector to reference angular accelerations for this frame
                    arot_diff = mod_arot[rm_nodes,:] - ref_arot[rm_nodes,:]
                    
                    h5dset = "/diff_angular_accelerations/frame_{}".format(timestep)
                    h5.create_dataset(h5dset, data = arot_diff, dtype=float)
                    fname = "Diff angular accelerations"
                    foo = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Vector"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset
                    
                    # Write relative error to reference accelerations of this frame
                    rel_errors = np.divide(np.linalg.norm(arot_diff, axis = 1), np.maximum(np.linalg.norm(ref_arot[rm_nodes,:], axis = 1), 1e-16 * np.ones((arot_diff.shape[0],))))
                    h5dset     = "/reldiff_angular_accelerations/frame_{}".format(timestep)
                    h5.create_dataset(h5dset, data = rel_errors, dtype=float)
                    fname = "Relative angular acceleration error to reference"
                    foo   = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":fname,"Type":"Scalar"})
                    ditem = etree.SubElement(foo,"DataItem",attrib={"DataType":"Float","Dimensions":"{0}".format(numnodes),
                                             "Format":"HDF","Precision":"8"})
                    ditem.text = h5file + ":" + h5dset

        del model_acc, mod_adisp
        if hasRot:
            del model_racc, mod_arot
            
        if reference_binout is not None:
            del ref_acc, ref_adisp
            if hasRot:
                del ref_racc, ref_arot
                
        
        for timestep, t in enumerate(timesteps_model):
            # Save stresses.
            pass
        
        
        # Finish the output
        h5.close()
        print("--> {0} {1} {2}".format(xdmffile, reference_xdmf, h5file))
        with open(xdmffile,"w") as outfile:
            outfile.write( '<?xml version="1.0" encoding="utf-8"?>\n' )
            outfile.write( '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n' )
            outfile.write( etree.tostring(xdmf,pretty_print=True).decode("utf-8") )
        
        with open(reference_xdmf,"w") as outfile:
            outfile.write( '<?xml version="1.0" encoding="utf-8"?>\n' )
            outfile.write( '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n' )
            outfile.write( etree.tostring(ref_xdmf,pretty_print=True).decode("utf-8") )

    def get_all_curve_ids(self):
        if self.curve_ids is not None:
            return self.curve_ids
            
        curvetickets = [self.tickets[i] for i in self.curveticketpos]
        curve_ids    = []
        
        # Find the correct part
        for ticket in curvetickets:
            for line in ticket:
                if line.startswith("*") or line.startswith("$"):
                    pass
                else:
                    # Read first line
                    lcont, _ = string_slices(line, [10])
                    lcont    = lcont[0]
                    curve_ids.append(int(lcont))
                    break
        
        self.curve_ids = curve_ids

        return curve_ids
    
    def write_reference_inputdeck(self, newfilename=None, output_K_ROM=False, dt_crv = None, tssfac_scl=1e20):
        output_tickets  = deepcopy(self.tickets)
        new_tickets     = []
        rm_elements     = []
        
        curve_ids       = self.get_all_curve_ids()
        if len(curve_ids) > 0:
            dt_crv_id       = min(999999, np.max(curve_ids) + 1)
            while dt_crv_id in curve_ids:
                dt_crv_id  -= 1
                if dt_crv_id == 0:
                    print("ERROR! Cannot create a new curve because all curve IDs are reserved by other curves.")
                    print("STOP INPUT DECK CREATION.")
                    return
        else:
            dt_crv_id = 1
        
        if newfilename is None:
            newfilename = self.filename.rsplit(".",1)[0] + "_reference.key"
        
        new_ctrl_timestep_written=False
        timestepticketpos = self.timestepticketpos
        if len(self.timestepticketpos) < 1:
            # insert a default *CONTROL_TIMESTEP ticket.
            print('Writing new *CONTROL_TIMESTEP card as none was present in the original model.')
            ctrl_dt_ticket = ["*control_timestep\n"]
            
            if dt_crv is None:
                dtinit = "{:.4e}".format(0.0)
                tsfstring = "{:.4e}".format(0.8)
                dt_crv_id_str = str(0)
            else:
                dtinit    = "{:.4e}".format(dt_crv[0,1])
                tsfstring = "{:.4e}".format(tssfac_scl)
                dt_crv_id_str = str(dt_crv_id)
            ctrl_dt_ticket.append("$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st\n")
            ctrl_dt_ticket.append("{:>10}".format(dtinit) + "{:>10}".format(tsfstring) +"         0     0.000     0.000" +  "{:>10}".format(dt_crv_id_str) + "         0         0\n")
            ctrl_dt_ticket.append("$#  dt2msf   dt2mslc     imscl    unused    unused     rmscl\n")
            ctrl_dt_ticket.append("     0.000         0         0                         0.000\n")
            output_tickets[1].extend(ctrl_dt_ticket)
            timestepticketpos = [1]
            new_ctrl_timestep_written=True
        
        with open(newfilename, "w+") as f:
            for i, ticket in enumerate(output_tickets):
                if i in self.eigvalticketpos:
                    # Skip the card when no intermittent eigenvalue analysis and matrix dumping is required.
                    continue
                
                if i == timestepticketpos[-1] and dt_crv is not None: # will crash if no CONTROL_TIMESTEP ticket is present.
                    if not new_ctrl_timestep_written:
                        for ic, c in enumerate(ticket):
                            if c.startswith("*") or c.startswith("$"):
                                continue
                            
                            # Only read and modify the first line
                            lcont, fixed_width = string_slices(c, [10,10,10,10,10,10,10,10])
                            lcont[0] = "{:.4e}".format(dt_crv[0,1])       # initial time step
                            lcont[1] = "{:.4e}".format(tssfac_scl)      # time step scaling factor to reach the upper bound specified by the curve
                            lcont[4] = "{:>10}".format(str(0))          # mass scaling turned off.
                            lcont[5] = "{:>10}".format(str(dt_crv_id))  # curve ID which controls the upper bound.
                            
                            if fixed_width:
                                ticket[ic] = "{:>10}".format(lcont[0])   + "{:>10}".format(lcont[1]) + "{:>10}".format(lcont[2]) + "{:>10}".format(lcont[3]) \
                                            + "{:>10}".format(lcont[4]) + "{:>10}".format(lcont[5]) + "{:>10}".format(lcont[6]) + "{:>10}".format(lcont[7])  \
                                            + '\n'
                            else:
                                ticket[ic] = ",".join(lcont) + '\n'
                            break
                        
                    # Create a new curve to control the time step after this ticket.
                    # A trick is employed to write the curve ticket into the same ticket object.
                    ticket.append("*define_curve\n")
                    ticket.append("$     LCID      SIDR       SFA       SFO      OFFA      OFFO     DATTYP\n")
                    ticket.append("{:>10}".format(str(dt_crv_id)) + "         0       1.0       1.0       0.0       0.0         0\n")
                    for r in range(dt_crv.shape[0]):
                        t  = dt_crv[r, 0]
                        dt = dt_crv[r, 1]
                        
                        tstring  = "{:.13e}".format(t)
                        dtstring = "{:.13e}".format(dt)
                        ticket.append("{:>20}".format(tstring) + "{:>20}".format(dtstring) + "\n")
                    
                    for ic, c in enumerate(ticket):
                        if c.startswith("*"):
                            f.write(c.upper())
                        else:
                            f.write(c)
                else:
                    for ic, c in enumerate(ticket):
                        if c.startswith("*database_nodout"):
                            nextline = ticket[ic+1]
                            j = 0
                            while nextline.startswith("$"): # skip comments
                                j+=1
                                nextline = ticket[ic+1+j]
                                
                            slices, fixed_width  = string_slices(nextline, [10, 10], sep=',')
                            if ("e" not in slices[0] and "E" not in slices[0]) and ("+" in slices[0] or "-" in slices[0]):
                                slices[0] = slices[0].replace('-', 'e-')
                                slices[0] = slices[0].replace('+', 'e+')
                                
                            frequency  = float(slices[0]) * 100
                            ticket[ic+1+j] = "{:.4e}".format(frequency) + ',' + slices[1] + '\n'
                        
                        if c.startswith("*"):
                            f.write(c.upper())
                        else:
                            f.write(c)
       
    def write_galerkin_inputdeck(self, newfilename=None, output_K_ROM=False, dt_crv = None, tssfac_scl=1e20):
        output_tickets  = deepcopy(self.tickets)
        new_tickets     = []
        rm_elements     = []
        
        curve_ids       = self.get_all_curve_ids()
        if len(curve_ids) > 0:
            dt_crv_id       = min(999999, np.max(curve_ids) + 1)
            while dt_crv_id in curve_ids:
                dt_crv_id  -= 1
                if dt_crv_id == 0:
                    print("ERROR! Cannot create a new curve because all curve IDs are reserved by other curves.")
                    print("STOP INPUT DECK CREATION.")
                    return
        else:
            dt_crv_id = 1
        
        if newfilename is None:
            newfilename = self.filename.rsplit(".",1)[0] + "_Galerkin.key"
        
        new_ctrl_timestep_written=False
        timestepticketpos = self.timestepticketpos
        if len(self.timestepticketpos) < 1:
            # insert a default *CONTROL_TIMESTEP ticket.
            print('Writing new *CONTROL_TIMESTEP card as none was present in the original model.')
            ctrl_dt_ticket = ["*control_timestep\n"]
            
            if dt_crv is None:
                dtinit = "{:.4e}".format(0.0)
                tsfstring = "{:.4e}".format(0.8)
                dt_crv_id_str = str(0)
            else:
                dtinit    = "{:.4e}".format(dt_crv[0,1])
                tsfstring = "{:.4e}".format(tssfac_scl)
                dt_crv_id_str = str(dt_crv_id)
            ctrl_dt_ticket.append("$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st\n")
            ctrl_dt_ticket.append("{:>10}".format(dtinit) + "{:>10}".format(tsfstring) +"         0     0.000     0.000" +  "{:>10}".format(dt_crv_id_str) + "         0         0\n")
            ctrl_dt_ticket.append("$#  dt2msf   dt2mslc     imscl    unused    unused     rmscl\n")
            ctrl_dt_ticket.append("     0.000         0         0                         0.000\n")
            output_tickets[1].extend(ctrl_dt_ticket)
            timestepticketpos = [1]
            new_ctrl_timestep_written=True

        with open(newfilename, "w+") as f:           
            for i, ticket in enumerate(output_tickets):
                if not output_K_ROM and i in self.eigvalticketpos:
                    # Skip the card when no intermittent eigenvalue analysis and matrix dumping is required.
                    continue
                
                if i == timestepticketpos[-1] and dt_crv is not None: # will crash if no CONTROL_TIMESTEP ticket is present.
                    if not new_ctrl_timestep_written:
                        for ic, c in enumerate(ticket):
                            if c.startswith("*") or c.startswith("$"):
                                continue
                            
                            # Only read and modify the first line
                            lcont, fixed_width = string_slices(c, [10,10,10,10,10,10,10,10])
                            lcont[0] = "{:.4e}".format(dt_crv[0,1])     # initial time step
                            lcont[1] = "{:.4e}".format(tssfac_scl)      # time step scaling factor to reach the upper bound specified by the curve
                            lcont[4] = "{:>10}".format(str(0))          # mass scaling turned off.
                            lcont[5] = "{:>10}".format(str(dt_crv_id))  # curve ID which controls the upper bound.
                            
                            if fixed_width:
                                ticket[ic] = "{:>10}".format(lcont[0])   + "{:>10}".format(lcont[1]) + "{:>10}".format(lcont[2]) + "{:>10}".format(lcont[3]) \
                                            + "{:>10}".format(lcont[4]) + "{:>10}".format(lcont[5]) + "{:>10}".format(lcont[6]) + "{:>10}".format(lcont[7])  \
                                            + '\n'
                            else:
                                ticket[ic] = ",".join(lcont) + '\n'
                            break
                            new_ctrl_timestep_written = True
                        
                    # Create a new curve to control the time step after this ticket.
                    # A trick is employed to write the curve ticket into the same ticket object.
                    ticket.append("*define_curve\n")
                    ticket.append("$     LCID      SIDR       SFA       SFO      OFFA      OFFO     DATTYP\n")
                    ticket.append("{:>10}".format(str(dt_crv_id)) + "         0       1.0       1.0       0.0       0.0         0\n")
                    for r in range(dt_crv.shape[0]):
                        t  = dt_crv[r, 0]
                        dt = dt_crv[r, 1]
                        
                        tstring  = "{:.13e}".format(t)
                        dtstring = "{:.13e}".format(dt)
                        ticket.append("{:>20}".format(tstring) + "{:>20}".format(dtstring) + "\n")
                    
                    for ic, c in enumerate(ticket):
                        if c.startswith("*"):
                            f.write(c.upper())
                        else:
                            f.write(c)
                else:    
                    for ic, c in enumerate(ticket):
                        if c.startswith("*database_nodout"):
                            nextline = ticket[ic+1]
                            j = 0
                            while nextline.startswith("$"): # skip comments
                                j+=1
                                nextline = ticket[ic+1+j]
                                
                            slices, fixed_width  = string_slices(nextline, [10, 10], sep=',')
                            if ("e" not in slices[0] and "E" not in slices[0]) and ("+" in slices[0] or "-" in slices[0]):
                                slices[0] = slices[0].replace('-', 'e-')
                                slices[0] = slices[0].replace('+', 'e+')
                                
                            frequency  = float(slices[0]) * 100
                            ticket[ic+1+j] = "{:.4e}".format(frequency) + ',' + slices[1] + '\n'
                        
                        if c.startswith("*"):
                            f.write(c.upper())
                        else:
                            f.write(c)

    # Function used to modify node data
    def modify_nodes(self, node_ids, node_data):

        updated_nodes = [self.extnid2int[i] for i in node_ids]
        if node_data.ndim == 1 or node_data.shape[1] == 1:
            self.temporalnodedata[updated_nodes,:] = node_data.reshape((-1,3))
        else:
            temporalnodedata = np.empty((self.node_count, 3, node_data.shape[1]))
            for time_step in range(node_data.shape[1]):
                temporalnodedata[:,:,time_step] = deepcopy(self.nodedata)
                temporalnodedata[updated_nodes,:,time_step] = node_data[:,time_step].reshape((-1,3))
            self.temporalnodedata = temporalnodedata


    def modify_timedata(self, time_data):
        self.timedata = time_data
                            

    def write_reduced_inputdeck(self, red_elem_set, contact_elem_set=[], lastECSWElement=1e11, newfilename=None, \
                                keep_all_nodes=True, node_reduction_range=[-9999999, 999999999], output_K_ROM=False, dt_crv = None,\
                                tssfac_scl = 1e20, rn_probe_nodes_ext=[]):
        '''
        Info: if an external EID is contained both in the reduced element set and in the contact set,
        make it part of the reduced element set (without MAT_NULL).
        
        INPUTS:
        - red_elem_set:           list or numpy array with the set of elements (external IDs) which are still part of the reduced mesh.
        - contact_elem_set:       list or numpy array with the set of elements (external IDs) which should be retained due to contact.
                                  These elements get shifted to MAT_NULL.
        - lastECSWElement:        last element which is part of the ECSW reduced region. Elements with larger external IDs will always be included.
        - newfilename:            String, specifying the new file name of the resulting key file.
        - keep_all_nodes:         Flag to control whether or not any nodes which are no longer needed should be removed from the mesh.
        - node_reduction_range:   Possibility to define a particular ID range in which the nodes should be removed (if nodes are removed at all)
        - output_K_ROM:           Flag to specify whether any eigenvalue tickets should be transferred to the reduced inputdeck as well.
                                  By default, eigenvalue key words are not transferred.
        - dt_crv:                 Time step curve. If a 2-column numpy array is given, then the curve defined by it is
                                  used to prescribe the time step size. First column: time, second column: time step size.
        - tssfact_scl:            scaling factor TSSFAC for increasing the time step size.
        - rn_probe_nodes_ext:     List which contains the external node IDs of any nodes which should not be removed from the input deck if the
                                  reduceNodes option is active. This means that these "probe" nodes are still present in the output of the 
                                  hyper-reduced RN-simulation and do not need to be reconstructed in a separate postprocessing step anymore.
        '''
        newfileappend = newfilename.rsplit(".", 1)[0]
        for deck in self.includes:
            # print(deck.filename.rsplit(".",1)[0] + newfileappend + ".key")
            filename = deck.filename.rsplit(".",1)[0] + newfileappend + ".key"
            deck.write_reduced_inputdeck(red_elem_set, contact_elem_set=contact_elem_set, lastECSWElement=lastECSWElement, newfilename=filename,\
                                         keep_all_nodes=keep_all_nodes, node_reduction_range=node_reduction_range, output_K_ROM=output_K_ROM, dt_crv = dt_crv,\
                                         tssfac_scl = tssfac_scl, rn_probe_nodes_ext=rn_probe_nodes_ext)
        output_tickets = deepcopy(self.tickets)

        include_list = [output_tickets[i] for i in self.includeticketpos]

        for ticket in include_list:
            ticket[1] = ticket[1].rsplit(".", 1)[0] + newfileappend + ".key\n"

        element_tickets = [output_tickets[i] for i in self.elementticketpos]
        new_tickets = []
        rm_elements = []
        omitted_elements_int = []
        
        curve_ids       = self.get_all_curve_ids()
        if len(curve_ids) > 0:
            dt_crv_id       = min(999999, np.max(curve_ids) + 1)
            while dt_crv_id in curve_ids:
                dt_crv_id  -= 1
                if dt_crv_id == 0:
                    print("ERROR! Cannot create a new curve because all curve IDs are reserved by other curves.")
                    print("STOP INPUT DECK CREATION.")
                    return
        else:
            dt_crv_id = 1
            
        # loop through all elements
        for ticket in element_tickets:
            try:
                int_eid = 0
                element_keyword_started = False
                for i, linecontent in enumerate(ticket):
                    # print("reading line %d" % i, ", content: " + linecontentecsw)
                    if element_keyword_started:
                        if (linecontent.startswith("*element_solid") or linecontent.startswith("*element_shell")):
                            # already processing an element keyword, can ignore the second keyword and move straight to the data
                            pass
                        elif linecontent.startswith("*"):
                            # new keyword which is not a element_shell or element_solid keyword!
                            element_keyword_started = False
                        elif linecontent.startswith("$"):
                            # ignore commented lines
                            pass
                        else:
                            ext_eid = self.elementdata[int_eid,0]
                            pid     = self.elementdata[int_eid,2]
                            
                            if ext_eid in red_elem_set or ext_eid > lastECSWElement:
                                # Element is part of the reduced set, do nothing.
                                rm_elements.append(int_eid)
                            
                            elif ext_eid in contact_elem_set:
                                rm_elements.append(int_eid)
                                
                                # Move the element to its corresponding shadow part with MAT_NULL
                                if pid not in self.shadowparts:
                                    # a shadow part for this PID has not been created yet. Create one now.
                                    shadow_part_ticket, shadow_mat_ticket = self.create_shadow_part(pid)
                                    new_tickets.append(shadow_part_ticket)
                                    new_tickets.append(shadow_mat_ticket)
                                    print('Created new shadow part and material for pid %d' % pid)
                                shadow_pid = self.shadowparts[pid]
                                
                                # replace pid by shadow_pid in the line content. To do that, first check if we have fixed width:
                                slices, fixed_width  = string_slices(linecontent, [8, 8], sep=',')
                                if fixed_width:
                                    shadow_pid_str = str(shadow_pid)
                                    ticket[i] = ticket[i][0:8] +  "{:>8}".format(shadow_pid_str) + ticket[i][16:]
                                else:
                                    slices[1] = str(shadow_pid)
                                    ticket[i] = ",".join(slices)
                            else:
                                # Comment out the element
                                omitted_elements_int.append(int_eid)
                                ticket[i] = "$ {}".format(ticket[i])
                            int_eid +=1
                    else:
                        # waiting for next *NODE keyword
                        if linecontent.startswith("*element_shell") or linecontent.startswith("*element_solid"):
                            element_keyword_started = True
                            
            except Exception as e:
                print("Error occurred in the following keyword, line #%d" %i)
                print(linecontent)
                print("Error: " + str(e))
                print("============= COMPLETE TICKET =============")
                print(ticket)
                print("Exiting inputdeck_writer.py after error.")
                return
        omitted_elements_ext = [self.elementdata[int_id][0] for int_id in omitted_elements_int]

        if not keep_all_nodes:
            rm_nodes, rm_nodes_ext, omitted_nodes_int, omitted_nodes_ext, new_node_tickets = self.reduce_nodes(rm_elements, node_reduction_range, output_tickets, kept_ext_nodes = rn_probe_nodes_ext)
            n_ticket_id = 0
            for i in self.nodeticketpos:
                output_tickets[i] = new_node_tickets[n_ticket_id]
                n_ticket_id += 1
            output_tickets = self.remove_nodes_from_sets(output_tickets, omitted_nodes_ext, omitted_elements_ext)
            
        # Insert new tickets
        insertion_position = self.partticketpos[-1]
        #print('Inserting new tickets at position %d' % insertion_position)
        
        if newfilename is None:
            newfilename = self.filename.rsplit(".",1)[0] + "_ecsw.key"
        
        new_ctrl_timestep_written=False
        timestepticketpos = self.timestepticketpos
        if len(self.timestepticketpos) < 1:
            # insert a default *CONTROL_TIMESTEP ticket.
            ctrl_dt_ticket = ["*control_timestep\n"]
            
            if dt_crv is None:
                dtinit = "{:.4e}".format(0.0)
                tsfstring = "{:.4e}".format(0.8)
                dt_crv_id_str = str(0)
            else:
                dtinit    = "{:.4e}".format(dt_crv[0,1])
                tsfstring = "{:.4e}".format(tssfac_scl)
                dt_crv_id_str = str(dt_crv_id)
            ctrl_dt_ticket.append("$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st\n")
            ctrl_dt_ticket.append("{:>10}".format(dtinit) + "{:>10}".format(tsfstring) +"         0     0.000     0.000" +  "{:>10}".format(dt_crv_id_str) + "         0         0\n")
            ctrl_dt_ticket.append("$#  dt2msf   dt2mslc     imscl    unused    unused     rmscl\n")
            ctrl_dt_ticket.append("     0.000         0         0                         0.000\n")
            output_tickets[1].extend(ctrl_dt_ticket)
            timestepticketpos = [1]
            new_ctrl_timestep_written=True
        
        with open(newfilename, "w+") as f:            
            for i, ticket in enumerate(output_tickets):
                if not output_K_ROM and i in self.eigvalticketpos:
                    # Skip the card when no intermittent eigenvalue analysis and matrix dumping is required.
                    continue
                
                if i == timestepticketpos[-1] and dt_crv is not None: # will crash if no CONTROL_TIMESTEP ticket is present.
                    if not new_ctrl_timestep_written:
                        for ic, c in enumerate(ticket):
                            if c.startswith("*") or c.startswith("$"):
                                continue
                            
                            # Only read and modify the first line
                            lcont, fixed_width = string_slices(c, [10,10,10,10,10,10,10,10])
                            lcont[0] = "{:.4e}".format(dt_crv[0,1])     # initial time step
                            lcont[1] = "{:.4e}".format(tssfac_scl)      # time step scaling factor to reach the upper bound specified by the curve
                            lcont[4] = "{:>10}".format(str(0))          # mass scaling turned off.
                            lcont[5] = "{:>10}".format(str(dt_crv_id))  # curve ID which controls the upper bound.
                            
                            if fixed_width:
                                ticket[ic] = "{:>10}".format(lcont[0])   + "{:>10}".format(lcont[1]) + "{:>10}".format(lcont[2]) + "{:>10}".format(lcont[3]) \
                                            + "{:>10}".format(lcont[4]) + "{:>10}".format(lcont[5]) + "{:>10}".format(lcont[6]) + "{:>10}".format(lcont[7]) + "\n"
                            else:
                                ticket[ic] = ",".join(lcont) + "\n"
                            break
                            new_ctrl_timestep_written = True
                        
                    # Create a new curve to control the time step after this ticket.
                    # A trick is employed to write the curve ticket into the same ticket object.
                    ticket.append("*define_curve\n")
                    ticket.append("$     LCID      SIDR       SFA       SFO      OFFA      OFFO     DATTYP\n")
                    ticket.append("{:>10}".format(str(dt_crv_id)) + "         0       1.0       1.0       0.0       0.0         0\n")
                    for r in range(dt_crv.shape[0]):
                        t  = dt_crv[r, 0]
                        dt = dt_crv[r, 1]
                        
                        tstring  = "{:.13e}".format(t)
                        dtstring = "{:.13e}".format(dt)
                        ticket.append("{:>20}".format(tstring) + "{:>20}".format(dtstring) + "\n")
                
                    for ic, c in enumerate(ticket):
                        if c.startswith("*"):
                            f.write(c.upper())
                        else:
                            f.write(c)
                else:
                    for ic, c in enumerate(ticket):
                        if c.startswith("*database_nodout"):
                            nextline = ticket[ic+1]
                            j = 0
                            while nextline.startswith("$"): # skip comments
                                j+=1
                                nextline = ticket[ic+1+j]
                                
                            slices, fixed_width  = string_slices(nextline, [10, 10], sep=',')
                            if ("e" not in slices[0] and "E" not in slices[0]) and ("+" in slices[0] or "-" in slices[0]):
                                slices[0] = slices[0].replace('-', 'e-')
                                slices[0] = slices[0].replace('+', 'e+')
                                
                            frequency  = float(slices[0]) * 100
                            ticket[ic+1+j] = "{:.4e}".format(frequency) + ',' + slices[1] + '\n'
                        
                        if c.startswith("*"):
                            f.write(c.upper())
                        else:
                            f.write(c)
                        
                if i == insertion_position:
                    for new_ticket in new_tickets:
                        for l in new_ticket:
                            f.write(l.upper())
        
        if not keep_all_nodes:
            return rm_nodes, rm_nodes_ext, omitted_nodes_int

    # returns all node data from a given inputdeck object
    def get_nodes(self):
        return self.nodedata, self.nodeids, self.extnid2int

    # returns all element data from a given inputdeck object
    def get_elements(self):
        return self.elementdata, self.ecsw_elementtypes, self.exteid2int

    # returns number of nodes in a given inputdeck
    def get_node_count(self):
        return self.node_count;

    # returns number of elements in a given inputdeck
    def get_element_count(self):
        return self.element_count;

    def ext2int(self, data, ext2int, rm=None, index=None):
        '''
        Info: function to return return specific data based on external node/element ids and a specific column index.
        
        INPUTS:
        - data:                   base data, typically self.nodedata or self.elementdata.
        - ext2int:                dictionary for mapping between external and internal nodes
        - rm:                     external node or element ids required
        - index:                  specific columns required from base data
        '''
        if index is None:
            index = range(len(ext2int[0]))
        if rm is None:
            return data[:,index]
        rm = [ext2int[i] for i in rm]
        data = data[rm, :]
        return data[:,index]

    # returns node ids of nodes in elements with eid in rm_elements and list of omitted nodes not in elements with eid in rm_elements
    def get_reduced_nodes(self, rm_element_list=None, node_range=None):
        '''
        
        INPUTS:
        rm_element_list:         list (array-like), containing the external IDs of the elements which belong to the reduced mesh.
        node_range:              tuple: (nid_min, nid_max), specifying the range of internal node IDs which are used for the reduction.
        
        OUTPUTS:
        node tuple:              node data related to external element ids passed in rm_element_list & external node ids related to
                                 external element ids passed in rm_element list
        omitted_nodes            external node ids of omitted nodes
        '''

        # list (array-like) of nodes in elements in rm_element_list
        rm_element_nodes = set(self.ext2int(self.elementdata, self.exteid2int, rm_element_list, range(3,11)).flat)
        if node_range is not None:
            nodeids = np.array(self.nodeids)
            below_range = nodeids < node_range[0] 
            above_range = nodeids > node_range[1]
            additional_nodes = nodeids[below_range]
            rm_element_nodes.update(additional_nodes)
            additional_nodes = nodeids[above_range]
            rm_element_nodes.update(additional_nodes)

        keys = set(self.extnid2int.keys())
        nodes = keys.intersection(rm_element_nodes)
        omitted_nodes = keys.difference(rm_element_nodes)
        rm_nodes = [self.extnid2int[i] for i in nodes]
        return (self.nodedata[rm_nodes,:], self.nodeids[rm_nodes]), omitted_nodes

    def self_reduce_nodes(self):
        rm_nodes = self.get_reduced_nodes()
        self.nodedata = rm_nodes[0][0]
        self.nodeids = rm_nodes[0][1]

    def remove_nodes_from_sets(self, output_tickets, omitted_nodes, omitted_elements):        
        '''
        Info: remove nodes from existing sets. currently only supports set_nodes and boundary_spc_nodes.
        
        INPUTS:
        - output_tickets:         deepcopy of self.tickets used for output
        OUTPUTS:
        - output_tickets:         modified output tickets
        '''

        # Getting set information from tickets
        # Remove nodes from *SET_ELEMENT_LIST & *SET_ELEMENT
        list_tickets = sorted(self.shellsetticketpos) 
        new_list_tickets = [deepcopy(output_tickets[i]) for i in list_tickets]

        for i, list_ticket in enumerate(new_list_tickets):
            list_elements = []
            new_list_ticket = []
            new_list_element = ""
            skipped = 0
            if "title" in list_ticket[0].lower():
                skip = 2
            else:
                skip = 1
            for list_element in list_ticket:
                if list_element.startswith("$"):
                    if skip >= 0:
                        skipped += 1
                    continue
                elif list_element.startswith("*"):
                    if skip >= 0:
                        skipped += 1
                    continue
                elif skip > 0:
                    skip -= 1
                    skipped += 1
                    continue
                else:
                    for num in string_slices(list_element, [10, 10, 10, 10, 10, 10, 10, 10])[0]:
                        try:
                            num = int(num)
                            if num > 0:
                                list_elements.append(num)
                        except:
                            pass
            list_elements = set(list_elements).difference(omitted_elements)
            j = 1
            for x in sorted(list_elements):
                new_list_element += str(x).rjust(10)
                if j%8 == 0:
                    new_list_ticket.append(new_list_element + "\n")
                    new_list_element = ""
                    j = 0
                j += 1
            if new_list_element != "":
                new_list_ticket.append(new_list_element + "\n")
            new_list_tickets[i] = list_ticket[:skipped] + new_list_ticket

        for i, j in enumerate(list_tickets):
            output_tickets[j] = new_list_tickets[i]

        # Remove nodes from *SET_NODE_LIST & *SET_NODE    
        list_tickets = sorted(self.nodesetticketpos) 
        new_list_tickets = [deepcopy(output_tickets[i]) for i in list_tickets]

        for i, list_ticket in enumerate(new_list_tickets):
            list_nodes = []
            new_list_ticket = []
            new_list_node = ""
            skipped = 0
            if "title" in list_ticket[0].lower():
                skip = 2
            else:
                skip = 1
            for list_node in list_ticket:
                if list_node.startswith("$"):
                    if skip >= 0:
                        skipped += 1
                    continue
                elif list_node.startswith("*"):
                    if skip >= 0:
                        skipped += 1
                    continue
                elif skip > 0:
                    skip -= 1
                    skipped += 1
                    continue
                else:
                    for num in string_slices(list_node, [10, 10, 10, 10, 10, 10, 10, 10])[0]:
                        try:
                            num = int(num)
                            if num > 0:
                                list_nodes.append(num)
                        except:
                            pass
            list_nodes = set(list_nodes).difference(omitted_nodes)
            j = 1
            for x in sorted(list_nodes):
                new_list_node += str(x).rjust(10)
                if j%8 == 0:
                    new_list_ticket.append(new_list_node + "\n")
                    new_list_node = ""
                    j = 0
                j += 1
            if new_list_node != "":
                new_list_ticket.append(new_list_node + "\n")
            new_list_tickets[i] = list_ticket[:skipped] + new_list_ticket

        for i, j in enumerate(list_tickets):
            output_tickets[j] = new_list_tickets[i]

        # Remove nodes from *SET_NODE_COLUMN
        col_tickets = self.nodecolticketpos
        new_col_tickets = [deepcopy(output_tickets[i]) for i in col_tickets]
        skipped = 0
        for col_ticket in new_col_tickets:
            if "title" in list_ticket[0].lower():
                skip = 2
            else:
                skip = 1
            for i, linecontent in enumerate(col_ticket):
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif skip > 0:
                    skip -= 1
                    continue
                elif int(string_slices(linecontent, [10, 10, 10, 10])[0][0]) in omitted_nodes:
                    col_ticket[i] = "$ {}".format(col_ticket[i])

        for i, j in enumerate(col_tickets):
            output_tickets[j] = new_col_tickets[i]

        # Remove nodes from *SET_NODE_GENERAL
        general_tickets = self.nodegeneralticketpos
        new_general_tickets = [deepcopy(output_tickets[i]) for i in general_tickets]
        for general_ticket in new_general_tickets:
            if "title" in list_ticket[0].lower():
                skip = 2
            else:
                skip = 1
            for i, linecontent in enumerate(general_ticket):
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif skip > 0:
                    skip -= 1
                    continue
                elif "NODE" in string_slices(linecontent, [10, 10])[0]:
                    new_line = "      NODE"
                    for node in string_slices(linecontent, [10, 10, 10, 10, 10, 10, 10, 10])[0]:
                        try:
                            if int(node) not in omitted_nodes:
                                new_line += node.rjust(10)
                        except:
                            pass
                    # print(new_line)
                    # print(general_ticket)
                    general_ticket[i] = new_line+"\n"

        for i, j in enumerate(general_tickets):
            output_tickets[j] = new_general_tickets[i]

        # Remove nodes from *BOUNDARY_SPC_NODES
        boundary_tickets = self.boundaryticketpos
        new_boundary_tickets = [deepcopy(output_tickets[i]) for i in boundary_tickets]

        for boundary_ticket in new_boundary_tickets:
            if "id" in boundary_ticket[0].lower():
                skip = 1
            else:
                skip = 0
            for i, linecontent in enumerate(boundary_ticket):
                if linecontent.startswith("$") or linecontent in ['\n', '\r\n']: 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif skip > 0:
                    skip -= 1
                    continue
                elif int(string_slices(linecontent, [10, 10, 10, 10])[0][0]) in omitted_nodes:
                    boundary_ticket[i] = "$ {}".format(boundary_ticket[i])

        for i, j in enumerate(boundary_tickets):
            output_tickets[j] = new_boundary_tickets[i]

        # Remove nodes from *INITIAL
        initial_tickets = self.initialpos
        new_initial_tickets = [deepcopy(output_tickets[i]) for i in initial_tickets]

        for initial_ticket in new_initial_tickets:
            for i, linecontent in enumerate(initial_ticket):
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif int(string_slices(linecontent, [10, 10])[0][0]) in omitted_nodes:
                    initial_ticket[i] = "$ {}".format(initial_ticket[i])

        for i, j in enumerate(initial_tickets):
            output_tickets[j] = new_initial_tickets[i]

        # Remove nodes from *CONSTRIAINED_LINEAR
        constrainedlinear_tickets = self.constrainedlinearpos
        new_constrainedlinear_tickets = [deepcopy(output_tickets[i]) for i in constrainedlinear_tickets]

        for constrained_ticket in new_constrainedlinear_tickets:
            for i, linecontent in enumerate(constrained_ticket):
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif i < 3:
                    continue
                elif int(string_slices(linecontent, [10, 10, 10, 10])[0][0]) in omitted_nodes:
                    constrained_ticket[i] = "$ {}".format(constrained_ticket[i])

        for i, j in enumerate(constrainedlinear_tickets):
            output_tickets[j] = new_constrainedlinear_tickets[i]

        # Remove nodes from *CONSTRAINED_SHELL
        constrainedshell_tickets = self.constrainedshellpos
        new_constrainedshell_tickets = [deepcopy(output_tickets[i]) for i in constrainedshell_tickets]

        for constrained_ticket in new_constrainedshell_tickets:
            for i, linecontent in enumerate(constrained_ticket):
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif int(string_slices(linecontent, [10, 10, 10, 10])[0][0]) in omitted_nodes:
                    constrained_ticket[i] = "$ {}".format(constrained_ticket[i])

        for i, j in enumerate(constrainedshell_tickets):
            output_tickets[j] = new_constrainedshell_tickets[i]

        # Remove nodes from *ELEMENT_MASS
        elementmass_tickets = self.elementmasspos
        new_elementmass_tickets = [deepcopy(output_tickets[i]) for i in elementmass_tickets]

        for elementmass_ticket in new_elementmass_tickets:
            for i, linecontent in enumerate(elementmass_ticket):
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif int(string_slices(linecontent, [8, 8, 8, 8])[0][1]) in omitted_nodes:
                    elementmass_ticket[i] = "$ {}".format(elementmass_ticket[i])

        for i, j in enumerate(elementmass_tickets):
            output_tickets[j] = new_elementmass_tickets[i]

        # Remove nodes from *ELEMENT_INERTIA
        isOmitted = False
        elementintertia_tickets = self.elementinertiapos
        new_elementintertia_tickets = [deepcopy(output_tickets[i]) for i in elementintertia_tickets]
        for elementinertia_ticket in new_elementintertia_tickets:
            if "offset" in elementinertia_ticket[0].lower():
                cards = 2
            else:
                cards = 1
            for i, linecontent in enumerate(elementinertia_ticket):
                
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                else:
                    lc = string_slices(linecontent, [8, 8, 8, 8])[0]
                    if lc[-1] == "":
                        if int(lc[1]) in omitted_nodes:
                            elementinertia_ticket[i] = "$ {}".format(elementinertia_ticket[i])
                            isOmitted = cards
                    else:
                        if isOmitted > 0:
                            elementinertia_ticket[i] = "$ {}".format(elementinertia_ticket[i])
                            isOmitted -= 1


        for i, j in enumerate(elementintertia_tickets):
            output_tickets[j] = new_elementintertia_tickets[i]

        return output_tickets
        

if __name__ == "__main__":
    run = True
    test = True

    if test:
        from Binout_reading import binout_reading
        import pickle
        from small_func import *

    if run:
        # deck = inputdeck("Main.key")
        # red_elem_set = [5]
        # deck.write_reduced_inputdeck(red_elem_set, newfilename="new.key", keep_all_nodes=False)

        deck = inputdeck("SFS_v6_ev.key")
        red_elem_set = list(range(25)) + [152832, 300013]
        deck.write_reduced_inputdeck(red_elem_set, newfilename="new_v6.key", keep_all_nodes=False)

        # name = "bumper.binout"
        # coordinates = binout_reading(name, False, 'coordinates')
        # coordinates = rearange_xyz(coordinates)
        # print(displacements.shape)
        # ids = binout_reading(name, False, 'ids')
        # timesteps = np.linspace(0, 1, num=201)
        # deck.modify_nodes(ids, coordinates)
        # deck.modify_timedata(timesteps)

        # with open("basis_vectors.pkl", "rb") as f:
        #     V = pickle.load(f)

        # deck.create_xdmf(Vdict = {"Basis" : V}, xdmffile="NEW.xdmf3", reduceNodes=False)

        
        # contact_elem = [50,51,52,53]
        
        # deck.write_reduced_inputdeck(red_elem_set, contact_elem_set=contact_elem, keep_all_nodes=False, node_reduction_range=[2,7])

        # print(nodedata.shape)
        # deck = inputdeck("Main.key")
        # red_elem_set = [5,6,7,8,9,10]
        # contact_elem = [50,51,52,53]
        # deck.write_reduced_inputdeck(red_elem_set, newfilename="1.key", keep_all_nodes=False)
        # deck.write_reduced_inputdeck(red_elem_set, contact_elem_set=contact_elem, keep_all_nodes=False, node_reduction_range=[2,7])
