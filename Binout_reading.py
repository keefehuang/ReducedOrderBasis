
# -*- coding: iso-8859-1 -*-
import numpy as np
from qd.cae.dyna import Binout




#########################################################################################################################################################

def binout_reading(binout_filename,normalize,data_type):
    
    '''
This function reads the nodout binary output file from lsdyna solver. It is compulsory to define in Lsdyna the output to be written in the bynary file
through the keywords DATABASE_NODOUT or DATABASE_ELOUT according to what it is needed. 
Then the displacements and the rotations for each coordinate are stored in separated snapshot matrices which rows are the nodes indexes, the columns being
the snapshot indexes.
    


    Arguments:
        @var binout_filename:           dictionary: {binout file name from lsdyna output} 
        @var normalization              dictionary: {'flag for normalization of snapshots' if normalize=='True' the snapshots are normalized}

    Outputs:
        @var Snapshot_x:                dictionary: {Snapshot matrix for x-displacements}
        @var Snapshot_y:                dictionary: {Snapshot matrix for y-displacements}
        @var Snapshot_z:                dictionary: {Snapshot matrix for z-displacements}
        @var Snapshot_rx:               dictionary: {Snapshot matrix for x-rotations}        
        @var Snapshot_ry:               dictionary: {Snapshot matrix for y-rotations}
        @var Snapshot_rz:               dictionary: {Snapshot matrix for z-rotations}  
    
    Libraries_required:
        qd.cae.dyna                     (Works only in Python 3)
        numpy
        
    
    '''  
    
    binout=Binout(binout_filename)
    var_displ=("x_displacement","y_displacement","z_displacement","rx_displacement","ry_displacement","rz_displacement")
    var_vel=("x_velocity","y_velocity","z_velocity","rx_velocity","ry_velocity","rz_velocity")
    var_acc=("x_acceleration","y_acceleration","z_acceleration","rx_acceleration","ry_acceleration","rz_acceleration")
    var_cor=("x_coordinate","y_coordinate","z_coordinate")
    var_t = ("time")

    if data_type == 'time':
        Snap = np.transpose(binout.read("nodout", var_t))
        return Snap

    if data_type=='coordinates + displacements + rvelocities':

            Snap1=np.vstack(
                (
                    np.transpose(binout.read("nodout",var_cor[0])),
                    np.transpose(binout.read("nodout",var_cor[1])),
                    np.transpose(binout.read("nodout",var_cor[2]))
                )
            )      
            Snap2=np.vstack(
                (
                np.transpose(binout.read("nodout",var_displ[0])),
                np.transpose(binout.read("nodout",var_displ[1])),
                np.transpose(binout.read("nodout",var_displ[2]))
                )
            )
            Snap3=np.vstack(
                (
                np.transpose(binout.read("nodout",var_vel[3])),
                np.transpose(binout.read("nodout",var_vel[4])),
                np.transpose(binout.read("nodout",var_vel[5]))
                )
            )
            if normalize==True:
                for i in range(0,Snap1.shape[1]):
            
                    if np.linalg.norm(Snap1[:,i])!=0:

                        Snap1[:,i]=Snap1[:,i]/np.linalg.norm(Snap1[:,i])
                        Snap2[:,i]=Snap2[:,i]/np.linalg.norm(Snap1[:,i])
                        Snap3[:,i]=Snap3[:,i]/np.linalg.norm(Snap1[:,i])
            return Snap1, Snap2, Snap3
 

    if data_type=='coordinates + displacements + velocities':

            Snap1=np.vstack(
                (
                    np.transpose(binout.read("nodout",var_cor[0])),
                    np.transpose(binout.read("nodout",var_cor[1])),
                    np.transpose(binout.read("nodout",var_cor[2]))
                )
            )      
            Snap2=np.vstack(
                (
                np.transpose(binout.read("nodout",var_displ[0])),
                np.transpose(binout.read("nodout",var_displ[1])),
                np.transpose(binout.read("nodout",var_displ[2]))
                )
            )
            Snap3=np.vstack(
                (
                np.transpose(binout.read("nodout",var_vel[0])),
                np.transpose(binout.read("nodout",var_vel[1])),
                np.transpose(binout.read("nodout",var_vel[2]))
                )
            )
            if normalize==True:
                for i in range(0,Snap1.shape[1]):
            
                    if np.linalg.norm(Snap1[:,i])!=0:

                        Snap1[:,i]=Snap1[:,i]/np.linalg.norm(Snap1[:,i])
                        Snap2[:,i]=Snap2[:,i]/np.linalg.norm(Snap1[:,i])
                        Snap3[:,i]=Snap3[:,i]/np.linalg.norm(Snap1[:,i])
            return Snap1, Snap2, Snap3
 

    if data_type=='coordinates + displacements':

            Snap1=np.vstack(
                (
                    np.transpose(binout.read("nodout",var_cor[0])),
                    np.transpose(binout.read("nodout",var_cor[1])),
                    np.transpose(binout.read("nodout",var_cor[2]))
                )
            )      
            Snap2=np.vstack(
                (
                np.transpose(binout.read("nodout",var_displ[0])),
                np.transpose(binout.read("nodout",var_displ[1])),
                np.transpose(binout.read("nodout",var_displ[2]))
                )
            )
            if normalize==True:
                for i in range(0,Snap1.shape[1]):
            
                    if np.linalg.norm(Snap1[:,i])!=0:

                        Snap1[:,i]=Snap1[:,i]/np.linalg.norm(Snap1[:,i])
                        Snap2[:,i]=Snap2[:,i]/np.linalg.norm(Snap1[:,i])
            return Snap1, Snap2

    if data_type=='coordinates':

            Snap=np.vstack((np.transpose(binout.read("nodout",var_cor[0])),np.transpose(binout.read("nodout",var_cor[1])),np.transpose(binout.read("nodout",var_cor[2]))))      

    if data_type=='displacements':
    
            Snap2=np.vstack(
                (
                np.transpose(binout.read("nodout",var_displ[0])),
                np.transpose(binout.read("nodout",var_displ[1])),
                np.transpose(binout.read("nodout",var_displ[2]))
                )
            )

    elif data_type=='rotations':     
    
            Snap=np.vstack((np.transpose(binout.read("nodout",var_displ[3])),np.transpose(binout.read("nodout",var_displ[4])),np.transpose(binout.read("nodout",var_displ[5]))))


    elif data_type=='displacements+rotations':
          
            Snap=np.vstack((np.transpose(binout.read("nodout",var_displ[0])),np.transpose(binout.read("nodout",var_displ[1])),np.transpose(binout.read("nodout",var_displ[2])),
                            np.transpose(binout.read("nodout",var_displ[3])),np.transpose(binout.read("nodout",var_displ[4])),np.transpose(binout.read("nodout",var_displ[5]))))

    elif data_type=='velocities':
    
            Snap=np.vstack((np.transpose(binout.read("nodout",var_vel[0])),np.transpose(binout.read("nodout",var_vel[1])),np.transpose(binout.read("nodout",var_vel[2]))))

    elif data_type=='angular_velocities':     
    
            Snap=np.vstack((np.transpose(binout.read("nodout",var_vel[3])),np.transpose(binout.read("nodout",var_vel[4])),np.transpose(binout.read("nodout",var_vel[5]))))


    elif data_type=='velocities+angular_velocities':
          
            Snap=np.vstack((np.transpose(binout.read("nodout",var_vel[0])),np.transpose(binout.read("nodout",var_vel[1])),np.transpose(binout.read("nodout",var_vel[2])),
                            np.transpose(binout.read("nodout",var_vel[3])),np.transpose(binout.read("nodout",var_vel[4])),np.transpose(binout.read("nodout",var_vel[5]))))

    elif data_type=='accelerations':
    
            Snap=np.vstack((np.transpose(binout.read("nodout",var_acc[0])),np.transpose(binout.read("nodout",var_acc[1])),np.transpose(binout.read("nodout",var_acc[2]))))

    elif data_type=='angular_accelerations':     
    
            Snap=np.vstack((np.transpose(binout.read("nodout",var_acc[3])),np.transpose(binout.read("nodout",var_acc[4])),np.transpose(binout.read("nodout",var_acc[5]))))


    elif data_type=='accelerations+angular_accelerations':
          
            Snap=np.vstack((np.transpose(binout.read("nodout",var_acc[0])),np.transpose(binout.read("nodout",var_acc[1])),np.transpose(binout.read("nodout",var_acc[2])),
                            np.transpose(binout.read("nodout",var_acc[3])),np.transpose(binout.read("nodout",var_acc[4])),np.transpose(binout.read("nodout",var_acc[5]))))

            
##Normalizing snapshots
    if normalize==True:
        for i in range(0,Snap.shape[1]):
            
            if np.linalg.norm(Snap[:,i])!=0:

                Snap[:,i]=Snap[:,i]/np.linalg.norm(Snap[:,i])

          
    return(Snap)






