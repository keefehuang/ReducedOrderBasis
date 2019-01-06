import os, sys
sys.path.append('include')

import h5py
import numpy as np
from copy import deepcopy
import subprocess
from lxml import etree
import re
import math





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
    __init__            : constructor, reads input file
    read_nodes          : reads and saves nodes of the model
    reduce_nodes        : creates alternative node tickets with the reduced nodes only
    read_elements       : reads and saves elements of the modes
    write_new_content   : writes new modified content to file

    modify_nodes        : modified internal nodes
    """

    def __init__(self, file, shadowoffset=1000, lastECSWElement=99999999):
        """ Constructor. Reads the input deck file and calls init_tickets().
        Input:  file name of an input deck (string)"""
        self.filename         = file
        self.file_content     = []
        self.file_content_new = []
        self.ecsw_elementtypes= []
        self.lastECSWElement  = lastECSWElement
        
        # ticket positions in self.tickets
        self.tickets          = []
        self.curveticketpos   = []
        self.nodeticketpos    = []
        self.elementticketpos = []
        self.partticketpos    = []
        self.matticketpos     = []
        self.shellsetticketpos  = []
        self.solidsetticketpos  = []
        self.eigvalticketpos  = []
        self.timestepticketpos= []
        self.initialpos       = []


        self.nodesetticketpos = []
        self.nodecolticketpos = []
        self.nodegeneralticketpos = []
        self.boundaryticketpos = []

        self.includeticketpos = []
        self.constrainedlinearpos = []
        self.constrainedshellpos = []
        self.initialpos       = []
        
        self.implicit_crvs    = []
        self.curve_ids        = None
        self.nodeids          = None
        self.node_count        = 0
        self.includes         = []
        self.extnid2int       = dict()
        self.exteid2int       = dict()
        self.nodedata         = None
        self.elementdata      = None
        self.part_contents    = {}
        self.mat_contents     = {}
        self.shadowparts      = {}
        self.shadowmats       = {}
        self.shadowoffset     = shadowoffset   # ID offset for newly created shadow parts and materials
        self.hasRigidWall     = False

        self.timedata         = None
        self.temporalnodedata = None

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
        self.temporalnodedata = deepcopy(self.nodedata)


    def read_nodes(self):
        """ Read NODE card of the input deck and save Node IDs and coordinates."""
        node_keyword_started = False
        self.node_count = 0
        node_and_include_position = sorted(self.nodeticketpos + self.includeticketpos)
        node_tickets = [self.tickets[i] for i in node_and_include_position]
        # Loop over all tickets
        for ticket in node_tickets:
            # First loop to count the nodes
            for i, linecontent in enumerate(ticket):
                if node_keyword_started:
                    if linecontent.startswith("*node"):
                        # already processing previous node keyword, can ignore the second keyword and move straight to the data
                        continue
                    elif linecontent.startswith("*include"):
                        # initialize an inputdeck instance for each include as it is reached during iteration
                        self.includes.append(inputdeck(ticket[1].strip()))
                        node_keyword_started = False
                    elif linecontent.startswith("*"):
                        # new keyword which is not a node keyword!
                        node_keyword_started = False
                        continue
                    elif linecontent.startswith("$"):
                        # ignore commented lines
                        pass
                    else:
                        # actual data is here.
                        self.node_count += 1
                else:
                    # waiting for next *NODE keyword
                    if linecontent.startswith("*node"):
                         node_keyword_started = True
                    elif linecontent.startswith("*include"):
                        # initialize an inputdeck instance for each include as it is reached during iteration
                        self.includes.append(inputdeck(ticket[1].strip()))

        for include in self.includes:
            self.node_count += include.get_node_count()


        self.nodedata = np.empty((self.node_count,3), dtype=float)
        self.nodeids  = np.empty(self.node_count, dtype=int)

        includes = iter(self.includes)

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
                    elif linecontent.startswith("*include"):
                        print("INCLUDES")
                        # upon reaching an include keyword, pull the stored node information and append i
                        new_include = next(includes)
                        include_node_data, include_node_ids, include_extnid2int = new_include.get_nodes()
                        include_node_count = new_include.get_node_count()
                        self.nodedata[int_nid:int_nid+include_node_count, :]  = include_node_data
                        self.nodeids[int_nid:int_nid+include_node_count, :]   = include_node_ids

                        offset = len(self.extnid2int)
                        for key, value in include_extnid2int.items():
                            include_extnid2int[key] = value + offset
                        self.extnid2int.update(include_extnid2int)

                        int_nid += include_node_count

                    else:
                        # Read and save actual data.
                        slices, _  = string_slices(linecontent, [8, 16, 16, 16], sep=',')                        
                        # if(int_nid > 50 and int_nid < 60):
                            # print('index %d, reading slice: %s.' % (int_nid, str(slices)))
                        ext_nid = int(slices[0])
                        x_coor  = float(slices[1])
                        y_coor  = float(slices[2])
                        z_coor  = float(slices[3])
                        # if(int_nid > 50 and int_nid < 60):
                        #     print(np.array([x_coor, y_coor, z_coor]))
                        self.nodedata[int_nid,:] = np.array([x_coor, y_coor, z_coor])
                        self.nodeids[int_nid]    = ext_nid
                        self.extnid2int[ext_nid] = int_nid
                        
                        int_nid +=1
                else:
                    # waiting for next *NODE keyword
                    if linecontent.startswith("*node"):
                         node_keyword_started = True
                    # upon reaching an include keyword, pull the stored node information and append i
                    elif linecontent.startswith("*include"):
                        new_include = next(includes)
                        include_node_data, include_node_ids, include_extnid2int = new_include.get_nodes()
                        include_node_count = new_include.get_node_count()
                        
                        self.nodedata[int_nid:int_nid+include_node_count, :]  = include_node_data
                        self.nodeids[int_nid:int_nid+include_node_count]   = include_node_ids

                        offset = len(self.extnid2int)
                        for key, value in include_extnid2int.items():
                            include_extnid2int[key] = value + offset
                        self.extnid2int.update(include_extnid2int)

                        int_nid += include_node_count
        
    def read_elements(self):
        """ Read ELEMENT cards of the input deck and save element IDs, nodes, and corresponding PIDs"""
        element_keyword_started = False
        self.element_count = 0
        element_and_include_position = sorted(self.elementticketpos + self.includeticketpos)
        element_tickets = [self.tickets[i] for i in element_and_include_position]
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
                        self.element_count += 1
                else:
                    # waiting for next *ELEMENT keyword
                    if linecontent.startswith("*element_shell") or linecontent.startswith("*element_solid") or linecontent.startswith("*element_seatbelt"):
                         element_keyword_started = True

        for new_include in self.includes:
            self.element_count += new_include.get_element_count()

        self.elementdata = np.empty((self.element_count,11), dtype=int)
        includes = iter(self.includes)

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
                    elif (eltype is "seatbelt" and linecontent.startswith("element_seatbelt")):
                        eltype = "seatbelt"
                    elif linecontent.startswith("*"):
                        # new keyword which is not a element keyword!
                        element_keyword_started = False
                    elif linecontent.startswith("$"):
                        # ignore commented lines
                        pass
                    elif linecontent.startswith("*include"):
                        new_include = next(includes)
                        include_element_data, include_ecsw_elementtypes, include_exteid2int = new_include.get_elements()
                        include_element_count = new_include.get_element_count()

                        self.elementdata[int_eid:int_eid+include_element_count, :] = include_element_data
                        for elmtyp in include_ecsw_elementtypes:
                            if elmtyp not in self.ecsw_elementtypes:
                                self.ecsw_elementtypes.append(elmtyp)

                        offset = len(self.exteid2int)
                        for key, value in include_exteid2int.items():
                            include_exteid2int[key] = value + offset
                        self.exteid2int.update(include_exteid2int)

                        int_eid += include_element_count

                    else:
                        # Read and save actual data.
                        if eltype is "shell":        # internal code: shell = 0
                            # 4 nodes
                            slices, _  = string_slices(linecontent, [8, 8, 8, 8, 8, 8], sep=',')
                            isTria = len(slices) <= 5
                            ext_eid = int(slices[0])                            
                            
                            if isTria:
                                self.elementdata[int_eid,0:11] = np.array([ext_eid, 0, slices[1], slices[2], slices[3], slices[4], slices[4], -1, -1, -1, -1])
                                if "SHELL3_ELFORM16" not in self.ecsw_elementtypes and ext_eid <= self.lastECSWElement:
                                    self.ecsw_elementtypes.append("SHELL3_ELFORM16")
                            else:
                                self.elementdata[int_eid,0:11] = np.array([ext_eid, 0, slices[1], slices[2], slices[3], slices[4], slices[5], -1, -1, -1, -1])
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
                    # elif linecontent.startswith("*element_seatbelt"):
                    #      element_keyword_started = True
                    #      eltype = "seatbelt"
                    elif linecontent.startswith("*element_beam"):
                         element_keyword_started = True
                         eltype = "beam"
                    elif linecontent.startswith("*include"):
                        new_include = next(includes)
                        include_element_data, include_ecsw_elementtypes, include_exteid2int = new_include.get_elements()
                        include_element_count = new_include.get_element_count()
                        self.elementdata[int_eid:int_eid+include_element_count, :] = include_element_data

                        offset = len(self.exteid2int)
                        for key, value in include_exteid2int.items():
                            include_exteid2int[key] = value + offset
                        self.exteid2int.update(include_exteid2int)

                        int_eid += include_element_count
                        
                        for elmtyp in include_ecsw_elementtypes:
                            if elmtyp not in self.ecsw_elementtypes:
                                self.ecsw_elementtypes.append(elmtyp)

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
        
    def reduce_nodes(self, rm_element_list, node_range):
        '''
        Reduces the nodes by removing any node which does not belong to an element
        
        INPUTS:
        rm_element_list:         list (array-like), containing the internal IDs of the elements which belong to the reduced mesh.
        node_range:              tuple: (nid_min, nid_max), specifying the range of internal node IDs which are used for the reduction.
        
        OUTPUTS:
        rm_nodes:                sorted numpy array, containing the internal node IDs of all nodes in the reduced mesh
        rm_nodes_ext:            sorted numpy array, containing the external ndoe IDs of all ndoes in the reduced mesh
        omitted_nodes            number of nodes that were removed from the mesh
        '''
        new_node_tickets = [deepcopy(output_tickets[i]) for i in self.nodeticketpos]
        rm_elements = self.elementdata[rm_element_list,:]
        rm_nodes = []
        rm_nodes_ext = []
        
        for int_nid, ext_nid in enumerate(self.nodeids):
            if ext_nid in rm_elements[:,3:] or (ext_nid < node_range[0] or ext_nid > node_range[1]):
                rm_nodes.append(int_nid)
                rm_nodes_ext.append(ext_nid)
                
        rm_nodes = self.nodedata[np.asarray(rm_nodes, dtype=int).sorted()]
        rm_nodes_ext = self.nodedata[np.asarray(rm_nodes_ext, dtype=int).sorted()]
        
        # loop through all node tickets
        omitted_nodes = 0
        for ticket in new_node_tickets:
            try:
                int_nid = 0
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
                            if int_nid not in rm_nodes:
                                # Comment out the node
                                ticket[i] = "$ {}".format(ticket[i])
                                omitted_nodes += 1
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
                        
        return rm_nodes, rm_nodes_ext, new_node_tickets, omitted_nodes

    
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
        else:
            rm_elements = output_eldata
            numnodes = self.nodedata.shape[0]

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
        
        # Create the PID field
        print('Writing PID field array')
        pidname = "PID"
        h5piddat = "/fields/{0}".format(pidname)
        h5.create_dataset(h5piddat,data=pid_vector,dtype=np.int32,compression="gzip")

        
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
        ecswname = "is_in_ECSW_region"
        h5ecswdat = "/fields/{0}".format(ecswname)
        h5.create_dataset(h5ecswdat,data=in_ECSW_region,dtype=np.int32,compression="gzip")
            
        # Append the basis vectors
        if Vdict is not None:
            print('Writing basis vectors for visualization')
            # iterate over the different provided bases
            bv_names = []
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
                        bv_names.append((h5file + ":" + h5propdat, basisvector.shape[0]))
                        h5.create_dataset(h5propdat,data=basisvector,dtype=float,compression="gzip")
                        
                else:
                    # V is a single column vector
                    basisvector = bv2matrix(V)
                    bv_name     = V_type + '_0'
                    
                    # Write the reshaped vector to the file
                    h5propdat = "/fields/{0}".format(bv_name)
                    h5.create_dataset(h5propdat,data=basisvector,dtype=float,compression="gzip")

                    bv_names.append((h5file + ":" + h5propdat, basisvector.shape[0]))

        if self.temporalnodedata.ndim > 2:
            timesteps = self.temporalnodedata.shape[2]
            if self.timedata is not None and len(self.timedata) != timesteps:
                timesteps = None
                print("Timestep data does not match number of timesteps in temporal node data")
        else:
            timesteps = 1
        
        for i, timestep in enumerate(range(timesteps)):
            grid = etree.SubElement(collection,"Grid",attrib={"Name":"frame {}".format(timestep)})
            if self.timedata is None:
                etree.SubElement(grid,"Time",attrib={"Value":"{0:.2e}".format(timestep)})
            else:
                etree.SubElement(grid,"Time",attrib={"Value":"{0:.2e}".format(self.timedata[i])})
            
            
            # Create XDMF node information
            # print('Processing XDMF node information, timestep {}\n'.format(timestep))
            h5dset = "/node_coordinates/frame_{}".format(timestep)

            try:
                data = self.temporalnodedata[:,:,timestep]
            except:
                data = self.temporalnodedata
            
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
                h5.create_dataset(h5dset,data=data[np.asarray(rm_nodes, dtype=int).sorted(),:],dtype=float)
            else:
                rm_elements = output_eldata
                numnodes = data.shape[0]
                h5.create_dataset(h5dset,data=data,dtype=float)
            
            ### Writing node position information
            geometry = etree.SubElement(grid,"Geometry",attrib={"Origin":"","Type":"XYZ"})
            ditem = etree.SubElement(geometry,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} {1}".format(numnodes,3),
                        "Format":"HDF","Precision":"8"})
            ditem.text = h5file + ":" + h5dset


            ### Writing topology information
            topo   = etree.SubElement(grid,"Topology",attrib={"Dimensions":str(nelemstot),"Type":"Mixed"})
            ditem = etree.SubElement(topo,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(ntopodat),
                                     "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5topodat

            ### Writing PID information
            pid = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":pidname,"Type":"Scalar"})
            ditem = etree.SubElement(pid,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                    "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5piddat


            ### Writing ECSW information
            ecsw = etree.SubElement(grid,"Attribute",attrib={"Center":"Cell","Name":ecswname,"Type":"Scalar"})
            ditem = etree.SubElement(ecsw,"DataItem",attrib={"DataType":"Int","Dimensions":"{0}".format(nelemstot),
                                    "Format":"HDF","Precision":"4"})
            ditem.text = h5file + ":" + h5ecswdat

            ### Writing basis vectors
            for basis_name in bv_names:
                basis = etree.SubElement(grid,"Attribute",attrib={"Center":"Node","Name":bv_name,"Type":"Vector"})
                ditem = etree.SubElement(basis,"DataItem",attrib={"DataType":"Float","Dimensions":"{0} 3".format(basis_name[1]),
                                        "Format":"HDF","Precision":"8"})
                ditem.text = basis_name[0]

                
        # Finish the output
        h5.close()
        print("--> {0} {1}".format(xdmffile,h5file))
        with open(xdmffile,"w") as outfile:
            outfile.write( '<?xml version="1.0" encoding="utf-8"?>\n' )
            outfile.write( '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n' )
            outfile.write( etree.tostring(xdmf,pretty_print=True).decode("utf-8") )



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
                                keep_all_nodes=True, node_reduction_range=[-math.inf, math.inf],output_K_ROM=False, dt_crv = None,\
                                tssfac_scl = 1e20):
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
        '''
        
        output_tickets = deepcopy(self.tickets)
        element_tickets = [output_tickets[i] for i in self.elementticketpos]
        new_tickets = []
        rm_elements = []
        int_elementid = []
        
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

        element_data = deepcopy(self.get_elements())
        for i, element in enumerate(element_data[0]):
            try:
                ext_eid = element[0]
                pid = element[2]
                if ext_eid in red_elem_set or ext_eid > lastECSWElement:
                    rm_elements.append(ext_eid)
                    int_elementid.append(i)
                elif ext_eid in contact_elem_set:
                    rm_elements.append(ext_eid)
                    int_elementid.append(i)
                    # Move the element to its corresponding shadow part with MAT_NULL
                    if pid not in self.shadowparts:
                        # a shadow part for this PID has not been created yet. Create one now.
                        shadow_part_ticket, shadow_mat_ticket = self.create_shadow_part(pid)
                        new_tickets.append(shadow_part_ticket)
                        new_tickets.append(shadow_mat_ticket)   
                        print('Created new shadow part and material for pid %d' % pid)
                    shadow_pid = self.shadowparts[pid]
                    
                    # replace pid by shadow_pid in the line content. To do that, first check if we have fixed width:
                    element[2] = str(shadow_pid)
                            
            except Exception as e:
                print("Error occurred in the following keyword, line #%d" %i)
                print(element)
                print("Error: " + str(e))
                print("============= COMPLETE TICKET =============")
                print(ticket)
                print("Exiting inputdeck_writer.py after error.")
                return
        
        if not keep_all_nodes:
            node_data, omitted_nodes = self.get_reduced_nodes(rm_element_list=rm_elements, node_range=node_reduction_range)
            rm_nodes = node_data[0]
            rm_nodes_ext = node_data[1]
            output_tickets = self.remove_nodes_from_sets(output_tickets, omitted_nodes)
        else:
            node_data = self.get_nodes()
            omitted_nodes = set([])
    

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

        nodes_appended = False
        with open(newfilename, "w+") as f:            
            for i, ticket in enumerate(output_tickets):
                if not output_K_ROM and i in self.eigvalticketpos:
                    # Skip the card when no intermittent eigenvalue analysis and matrix dumping is required.
                    continue                
                if i == timestepticketpos[-1] and dt_crv is not None: # will crash if no CONTROL_TIMESTEP ticket is present.                    if not new_ctrl_timestep_written:
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
                    lc_line = "".join(ticket).lower()
                    if "*include" in lc_line:
                        continue
                    if nodes_appended == False:
                        if "*node" in lc_line or "*element_shell" in lc_line or "*element_solid" in lc_line:
                            node_element_ticket = node_data_to_ticket(node_data) + element_data_to_ticket(element_data, rm_elems=int_elementid)
                            f.write(node_element_ticket)
                            nodes_appended = True
                            continue
                    elif "*node" in lc_line or "*element_shell" in lc_line or "*element_solid" in lc_line:
                        continue
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
            return rm_nodes, rm_nodes_ext, omitted_nodes

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

    def remove_nodes_from_sets(self, output_tickets, omitted_nodes):        
        '''
        Info: remove nodes from existing sets. currently only supports set_nodes and boundary_spc_nodes.
        
        INPUTS:
        - output_tickets:         deepcopy of self.tickets used for output
        OUTPUTS:
        - output_tickets:         modified output tickets
        '''

        # Getting set information from tickets

        # Remove nodes from $SET_NODE_LIST & *SET_NODE    
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
                    if skip > 0:
                        skipped += 1
                    continue
                elif list_node.startswith("*"):
                    if skip > 0:
                        skipped += 1
                    continue
                elif skip > 0:
                    skip -= 1
                    skipped += 1
                    continue
                else:
                    for num in string_slices(list_node, [10, 10, 10, 10, 10, 10, 10, 10])[0]:
                        try:
                            list_nodes.append(int(num))
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

        # Remove nodes from $SET_NODE_COLUMN
        col_tickets = self.nodecolticketpos
        new_col_tickets = [deepcopy(output_tickets[i]) for i in col_tickets]
        skipped = 0
        for col_ticket in new_col_tickets:
            if "TITLE" in list_ticket:
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

        # Remove nodes from $SET_NODE_GENERAL
        general_tickets = self.nodegeneralticketpos
        new_general_tickets = [deepcopy(output_tickets[i]) for i in general_tickets]
        for general_ticket in new_general_tickets:
            if "TITLE" in list_ticket:
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

        boundary_tickets = self.boundaryticketpos
        new_boundary_tickets = [deepcopy(output_tickets[i]) for i in boundary_tickets]

        for boundary_ticket in new_boundary_tickets:
            for i, linecontent in enumerate(boundary_ticket):
                if linecontent.startswith("$"): 
                    continue
                elif linecontent.startswith("*"):
                    continue
                elif int(string_slices(linecontent, [10, 10, 10, 10])[0][0]) in omitted_nodes:
                    boundary_ticket[i] = "$ {}".format(boundary_ticket[i])

        for i, j in enumerate(boundary_tickets):
            output_tickets[j] = new_boundary_tickets[i]

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

        return output_tickets
        

if __name__ == "__main__":
    run = True
    test = True

    if test:
        from Binout_reading import binout_reading
        import pickle
        from small_func import *

    if run:
        deck = inputdeck("SFS_Dyna_main.key")

        name = "bumper.binout"
        coordinates = binout_reading(name, False, 'coordinates')
        coordinates = rearange_xyz(coordinates)
        # print(displacements.shape)
        ids = binout_reading(name, False, 'ids')
        timesteps = np.linspace(0, 1, num=201)
        deck.modify_nodes(ids, coordinates)
        deck.modify_timedata(timesteps)

        with open("basis_vectors.pkl", "rb") as f:
            V = pickle.load(f)

        deck.create_xdmf(Vdict = {"Basis" : V}, xdmffile="NEW.xdmf3", reduceNodes=False)

        # print(nodedata.shape)
        # deck = inputdeck("Main.key")
        # red_elem_set = [5,6,7,8,9,10]
        # contact_elem = [50,51,52,53]
        # deck.write_reduced_inputdeck(red_elem_set, newfilename="1.key", keep_all_nodes=False)
        # deck.write_reduced_inputdeck(red_elem_set, contact_elem_set=contact_elem, keep_all_nodes=False, node_reduction_range=[2,7])
