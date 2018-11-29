
# coding: utf-8

# # Preprocessing linefinder Data for Display in Firefly

# This notebook converts and post-processes data created with [linefinder](https://github.com/zhafen/linefinder) into a format readable by [Firefly](https://github.com/ageller/Firefly).

# To get started, you will need to install linefinder via
# ```
# pip install linefinder --user
# ```
# Note that, as of writing, linefinder is not yet Python3 enabled... It will be updated soon.

# Most of the things you, as the user, will change are in the [Parameters](#Parameters) or [Custom Data](#User-Defined-Data,-Classifications,-and-Filters) section, just below the imports.

# ## Imports

# In[1]:


import copy
import numpy as np
import os
import h5py
import shutil


# In[2]:


from linefinder_FIREreader import FIREreader


# In[3]:


import linefinder.analyze_data.worldlines as a_worldlines


# In[4]:


import linefinder.utils.presentation_constants as p_constants
import linefinder.utils.file_management as file_management
import linefinder.config as linefinder_config


# In[5]:


import galaxy_dive.utils.executable_helpers as exec_helpers
import galaxy_dive.analyze_data.particle_data as particle_data
import galaxy_dive.utils.data_operations as data_operations


# ## Parameters

# In[6]:


# What type of linefinder data are we working with? Galaxy focused? CGM focused?
analysis_type = 'CGM'


# In[7]:


# What simulation to run this for?
sim_name = 'm10y'

# What snapshot to run this for?
snum = 172


# In[8]:


# How do we define our galaxies?
galdef = '_galdefv3'

# This is used to index the halo data,
# and should usually be the number of snapshots in the simulation
ahf_index = 600


# In[9]:


# What Filters to Use
filter_data_keys = [ 'T', 'Z', 'Den', 'is_in_main_gal', 'is_in_other_gal', 'PType' ]

# These filters should be in log space
log_filters = [ 'T', 'Z', 'Den', ]


# In[10]:


# Display parameters
time_data = False # If True, plot worldlines for particles
n_displayed = 100 # Number of worldlines displayed for the time data

# If True, t=0 is the snapshot at which particles were selected (assumes CGM data)
center_time_on_snapshot = True

# If True, display a 100x100x100 pkpc ruler with 1 kpc spacing.
include_ruler = True 

# If True, display a circle with radius R_gal aligned parallel to the angular momentum of the stars,
# and a 50 pkpc long line aligned with the angular momentum of the stars (with 1kpc spacing)
include_disk = True

# If True, add filters that allow us to filter for how long a particle has spent as a
# given classification
time_filters = False

# If True, display only particles that have left their source 0.5-1 Gyr ago
include_only_recent_data = False

# If True, only select gas particles
only_select_gas = True


# In[11]:


# Save directory
if time_data:
    target_dir = '/home1/03057/zhafen/repos/CGM-origins-pathlines/data'
else:
    target_dir = '/home1/03057/zhafen/repos/CGM-origins/data'


# In[12]:


# What classifications to use, from the set of classification available in linefinder
if analysis_type == 'CGM':
    classification_list = p_constants.CLASSIFICATIONS_CGM_ORIGIN
#     classification_list.append( 'is_in_galaxy_halo_interface' )
elif analysis_type == 'galaxy':
    classification_list = p_constants.CLASSIFICATIONS_A


# In[13]:


# Labels
classification_labels = {
    'is_in_CGM' : 'All',
    'is_CGM_IGM_accretion' : 'IGMAcc',
    'is_CGM_wind' : 'Wind',
    'is_CGM_satellite_wind' : 'SatWind',
    'is_CGM_satellite_ISM' : 'SatISM',
    'is_CGM_NEP' : 'IGMAcc',
    'is_CGM_IP' : 'Wind',
    'is_CGM_EP' : 'SatWind',
    'is_CGM_satellite' : 'SatISM',
    'is_in_galaxy_halo_interface' : 'Interface',
    'is_fresh_accretion' : 'FreshAcc',
    'is_NEP_wind_recycling' : 'NEPWindRecycling',
    'is_mass_transfer' : 'Transfer',
    'is_merger_gas' : 'ISMMerger',
    'is_merger_star' : 'StarMerger',
}
filter_labels = {
    'T' : 'TemperatureK',
    'Z' : 'MetallicitySolar',
    'Den' : 'DensityPerCC',
    'is_in_main_gal' : 'InMainGal',
    'is_in_other_gal' : 'InOtherGal',
    'PType' : 'PType',
    'R' : 'RadiusPkpc',
    'particle_ind' : 'ParticleID',
    'time' : 'TimeGyr',
    'time_as_CGM_NEP' : 'TimeSinceAccGyr',
    'time_as_outside_any_gal_IP' : 'TimeSinceEjectionGyr',
    'time_as_outside_any_gal_EP' : 'TimeSinceEjectionGyr',
}


# ## User-Defined Data, Classifications, and Filters
# A classification is a set of particles you want to show up as a completely different color in Firefly. When `time_data == True` you'll also show `n_displayed` worldlines for particles that are classified as your classification at the specified snapshot `snum`.
# 
# Put your custom classifications below, in the form of boolean arrays where `True` means the particle is part of the classification. The arrays need to be the shape of the particle data, usually `(1e5, 600)`. If you have a classification for a particular particle that's true throughout its entire history, just tile the data.
# 
# A custom filter will allow you to filter the values for a given classification

# By default the visualization will currently use data from `/scratch/03057/zhafen/linefinder_data` and halo data from `/scratch/03057/zhafen/core`. If you would like to change these permanently you can do so in `linefinder/config.py`. If you used `pip` to install linefinder, then this is probably located in `~/.local/lib/python2.7/site-packages/linefinder`. Otherwise you can overwrite the defaults below.

# In[14]:


custom_data_dir = None
custom_halo_data_dir = None


# In[15]:


# Add any classifications you would like to this dictionary,
# and they'll be added to the visualization
custom_classifications = {
#     'custom' : np.random.randint( 0, 2, size=(100000, 600) ).astype( bool ),
}


# In[16]:


# Add any custom filters you would like to this dictionary,
# and they'll be added to the visualization
custom_filters = {
#     'custom' : 10.**np.random.normal( 10., 5., size=(100000, 600) ),
}


# In[17]:


# Choose colors for your custom classifications, in RGB
classification_colors = {
    'custom' : [ 1., 1., 0. ],
}
# Add labels for your classifications and filters. Labels cannot contain spaces or any special
# characters, unfortunately
custom_classification_labels = {
    'custom' : 'Custom',
}
custom_filter_labels = {
    'custom' : 'CustomFilter',
}

# If you want any of your filters to be in logspace, indicate so here
log_custom_filters = [ 'custom' ]


# ## Load Data

# In[18]:


# If using the commandline, change the parameters
sim_name, snum = exec_helpers.choose_config_or_commandline(
    [ sim_name, snum ]
)


# In[19]:


tag_tails = {
    'CGM' : '_CGM_snum{}'.format( snum ),
    'galaxy' : '',
}
tag_tail = tag_tails[analysis_type]


# In[20]:


print( "Running for {}, snapshot {}".format( sim_name, snum ) )


# In[21]:


# Load the a helper for loading files easily
file_manager = file_management.FileManager( project='CGM_origin' )


# In[22]:


ind = ahf_index - snum


# In[23]:


defaults, variations = file_manager.get_linefinder_analysis_defaults_and_variations(
    tag_tail,
    sim_names = [ sim_name ],
    galdef = galdef,
)


# In[24]:


if sim_name == 'm12i':
    used_args = defaults
else:
    used_args = variations[sim_name]


# In[25]:


# Custom data dirs, if given.
if custom_data_dir is not None:
    used_args[data_dir] = custom_data_dir
if custom_halo_data_dir is not None:
    used_args[halo_data_dir] = custom_halo_data_dir


# In[26]:


w = a_worldlines.Worldlines( **used_args )


# In[27]:


# Wind Velocity Filter

# Fresh start
w.data_masker.clear_masks( True )

# Don't select particles past our classification snapshot
w.mask_data( 'snum', 1, snum, tile_data=True )

def n_snum_mask( n_key ):
    '''Returns a mask for particles that have had n change since snum.'''
    
    n_snum = w.get_data(
        n_key,
        sl = (slice(None), ind),
    )

    n_tiled = np.tile( n_snum, ( w.n_snaps, 1 ) ).transpose()

    mask = w.get_data( n_key ) != n_tiled
    
    return mask

# Mask data that has left or entered the galaxy a different number of times
w.mask_data(
    'n_in_mismatch',
    custom_mask = n_snum_mask( 'n_in' ),
    optional_mask = True,
)

# Get wind velocities out
max_vel_after_in = w.get_selected_data(
    'Vr',
    optional_masks=[ 'n_in_mismatch' ],
    compress = False,
    scale_key = 'Vmax',
).max( axis=1 )

# Add to custom filters and labels
custom_filters['max_vel_to_prev_acc'] = np.tile(
    max_vel_after_in.filled( fill_value=-1. ),
    ( w.n_snaps, 1 ),
).transpose()
custom_filter_labels['max_vel_to_prev_acc'] = 'MaxVelToPrevAcc'

# Reclear masks
w.data_masker.clear_masks( True )


# ### Rest of Data Loading

# In[28]:


# Include custom classifications
for c_c in custom_classifications.keys():
    
    w.data[c_c] = custom_classifications[c_c]
    
    classification_list.append( c_c )
    
    classification_labels[c_c] = custom_classification_labels[c_c]


# In[29]:


# Include custom filters
for c_f in custom_filters.keys():
    
    w.data[c_f] = custom_filters[c_f]
    
    filter_data_keys.append( c_f )
    
    filter_labels[c_f] = custom_filter_labels[c_f]
    
    if c_f in log_custom_filters:
        log_filters.append( c_f )


# In[30]:


def get_data( data_key, classification, time_data, seed=None, *args, **kwargs ):

    if time_data:
        
        if include_only_recent_data and ( classification in time_keys.keys() ):
            optional_masks = [ time_keys[classification], ]
        else:
            optional_masks = []

        return w.get_selected_data_over_time(
            data_key,
            snum = snum,
            classification = classification,
            n_samples = n_displayed,
            seed = seed,
            optional_masks = optional_masks,
            *args, **kwargs
        ).flatten()
        
    else:
       
        return w.get_selected_data(
            data_key,
            sl = (slice(None),ind),
            classification = classification,
            *args, **kwargs
        )


# ## Mask Data

# In[31]:


time_keys = {
    'is_CGM_NEP' : 'time_as_CGM_NEP',
    'is_CGM_IP' : 'time_as_outside_any_gal_IP',
    'is_CGM_EP' : 'time_as_outside_any_gal_EP',
}


# In[32]:


if include_only_recent_data:
    for classification in classification_list:
        if classification in time_keys.keys():
            w.data_masker.mask_data(
                time_keys[classification],
                0.5,
                1.0,
                optional_mask = True,
                mask_name = time_keys[classification],
            )


# In[33]:


if only_select_gas:
    w.data_masker.mask_data(
        'PType',
        data_value=0,
    )


# ## Data Retrieval

# In[34]:


# Set up random seeds (useful for sampling consistent particles
seeds = {}
for classification in classification_list:
    seeds[classification] = np.random.randint( 1e7 )


# In[35]:


# Remove classifications where we have no data with that classification
for classification in copy.copy( classification_list ):
    n_class = w.get_selected_data(
        classification,
        tile_data = True,
        compress = False
    )[:,ind].sum()
    
    if n_class == 0:
        classification_list.remove( classification )


# In[36]:


positions = {}
velocities = {}

# Get positions and velocities for all classifications we care about
for classification in classification_list:
    
    print( classification )

    # Get positions and velocities
    class_pos = []
    class_vel = []
        
    for pos_key, vel_key  in zip( [ 'Rx', 'Ry', 'Rz' ], [ 'Vx', 'Vy', 'Vz' ] ):
            ri = get_data(
                pos_key,
                classification = classification,
                time_data = time_data,
                seed = seeds[classification],
            )
            class_pos.append( ri )

            vi = get_data(
                vel_key,
                classification = classification,
                time_data = time_data,
                seed = seeds[classification],
            )
            class_vel.append( vi )
            
    # Make into a numpy array
    class_pos = np.array( class_pos ).transpose()
    class_vel = np.array( class_vel ).transpose()
    
    # Store the data
    positions[classification] = class_pos
    velocities[classification] = class_vel


# In[37]:


# In the case of time data, we also include particle ind
if time_data:
    filter_data_keys.append( 'particle_ind' )


# In[38]:


# Filter Data
filter_data = {}
for filter_key in filter_data_keys:
        
    filter_data[filter_key] = {}
    
    for classification in classification_list:

        filter_data[filter_key][classification] = get_data(
            filter_key,
            classification = classification,
            time_data = time_data,
            seed = seeds[classification],
        )


# In[39]:


# Include a time array
if time_data:
    filter_key = 'time'
    filter_data[filter_key] = {}
    
    for classification in classification_list:
        
        time_arr = get_data(
            filter_key,
            classification = classification,
            time_data = time_data,
            seed = seeds[classification],
            tile_data = True,
        )
        
        if center_time_on_snapshot:
            time_arr -= w.data['time'][ind]
        
        filter_data[filter_key][classification] = time_arr


# In[40]:


if time_filters:    
    for classification in classification_list:
        
        if classification not in time_keys.keys():
            continue
            
        filter_key = time_keys[classification]
        
        time_filter_arr = get_data(
            filter_key,
            classification = classification,
            time_data = time_data,
            seed = seeds[classification],
        )
        
        try:
            filter_data[filter_key][classification] = time_filter_arr
        except KeyError:
            filter_data[filter_key] = {}
            filter_data[filter_key][classification] = time_filter_arr


# In[41]:


# Setup a ruler
if include_ruler:
    
    n_ruler_points = 101
    x = np.array([
        np.linspace( 0., 100., n_ruler_points ),
        np.zeros( n_ruler_points ),
        np.zeros( n_ruler_points ),
    ]).transpose()
    
    y = np.roll( x, 1 )
    z = np.roll( x, 2 )
    
    positions['ruler'] = np.concatenate( [ x, y, z ] )


# In[42]:


# Setup disk
if include_disk:
    
    # Load data
    s_data = particle_data.ParticleData(
        sdir = file_manager.get_sim_dir( sim_name ),
        snum = snum,
        ptype = linefinder_config.PTYPE_STAR,
        halo_data_dir = file_manager.get_halo_dir( sim_name ),
        main_halo_id = linefinder_config.MAIN_MT_HALO_ID[sim_name],    
    )
    
    # Get length scale
    r_gal = w.r_gal[ind]

    # Create cicle to rotate
    circle = []
    for phi in np.linspace( 0., 2.*np.pi, 256 ):

        circle.append(
            [ r_gal*np.cos(phi), r_gal*np.sin(phi), 0. ]
        )

    circle = np.array( circle )

    ang_mom_vec = s_data.total_ang_momentum / np.linalg.norm( s_data.total_ang_momentum )
    disk_pos = data_operations.align_axes( circle, ang_mom_vec, align_frame=False )

    # Get axis pos
    ang_mom_pos = np.tile( ang_mom_vec, (51,1) )
    axis_pos = np.linspace( 0., 50., 51 )[:,np.newaxis]*ang_mom_pos
    
    positions['disk'] = np.concatenate( [ disk_pos, axis_pos ])


# ## Load and read in the FIRE Data

# In[43]:


## initialize reader object and choose simulation to run
reader = FIREreader()
reader.directory = file_manager.get_sim_dir( sim_name )
reader.snapnum = snum
## could read this from snapshot times
current_redshift = w.redshift[snum]


# In[44]:


## decide which part types to save to JSON
reader.returnParts = [ 'PartType1', 'PartType4', ]
originalReturnParts = copy.copy( reader.returnParts )

## choose the names the particle types will get in the UI
reader.names = {
    'PartType0':'Gas', 
    'PartType1':'HRDM', 
    'PartType2':'LRDM', 
    'PartType4':'Stars',
}


# In[45]:


# Add Linefinder categories
for classification in classification_list:
    reader.returnParts.append( classification )
    reader.names[classification] = classification_labels[classification]


# In[46]:


# Add Ruler and Disk
if include_ruler:
    reader.returnParts.append( 'ruler' )
    reader.names['ruler'] = 'ruler'
if include_disk:
    reader.returnParts.append( 'disk' )
    reader.names['disk'] = 'disk'


# In[47]:


#define the defaults; this must be run first if you want to change the defaults below
reader.defineDefaults()

## by what factor should you sub-sample the data (e.g. array[::decimate])
decimate = [ 200, ] * len( originalReturnParts )
for i in range( len( classification_list) ):
    decimate.append( 1 )


# In[48]:


if include_ruler:
    decimate.append( 1 )
if include_disk:
    decimate.append( 1 )


# In[49]:


## load in the data from hdf5 files and put it into reader.partsDict
for i,p in enumerate(reader.returnParts):
    reader.decimate[p] = decimate[i]
    reader.returnKeys[p] = []#['Coordinates', 'Density','Velocities','HIIAbundance','Temperature','AgeGyr']
    #Note: you should only try to filter on scalar values (like density).  
    #The magnitude of the Velocities are calculated in Firefly, and you will automatically be allowed to filter on it
    reader.addFilter[p] = []#[False, True, False,True,True,True]
    ## tell it to do the log of density when filtering
    reader.dolog[p] = []#[False, True, False,False,True,False]
    
    
    #NOTE: all dictionaries in the "options" reference the swapped names (i.e., reader.names) you define above.  
    #If you don't define reader.names, then you can use the default keys from the hdf5 files 
    #(but then you will see those hdf5 names in the Firefly GUI)
    pp = reader.names[p]
    ## set the initial size of the particles when the interface loads
    reader.options['sizeMult'][pp] = 1.
 


# In[50]:


# Adjust Point Sizes
for classification in classification_list:
    
    pp = classification_labels[classification]
    
    # Adjustments in some cases to allow easier visibility of all categories
    if time_data:
        reader.options['sizeMult'][pp] = 2.5
    else:
#         if classification == 'is_CGM_NEP':
#             reader.options['sizeMult'][pp] = 2.
#         elif classification == 'is_CGM_IP':
#             reader.options['sizeMult'][pp] = 3.
#         elif classification == 'is_in_CGM':
#             reader.options['sizeMult'][pp] = 2.5
#         else:
        reader.options['sizeMult'][pp] = 3.5
            
if include_ruler:
    reader.options['sizeMult']['ruler'] = 4.
if include_disk:
    reader.options['sizeMult']['disk'] = 4.


# In[51]:


# Set the default colors when the interface loads
reader.options['color'] = {
    'Gas':  [1., 0., 0., 1.],
   'HRDM': [1., 1., 0., 0.1],  
   'LRDM': [1., 1., 0., 0.1],  
   'Stars': [ 232./360., 221.21/360., 16.24/360., 0.1],
} 

for classification in classification_list:
    if classification in classification_colors.keys():
        color = classification_colors[classification]
    elif classification is not 'is_in_CGM':
        color = list( p_constants.CLASSIFICATION_COLORS_B[classification] )
    else:
        color = [ 1., 1., 1. ]
    color.append( 1. )
    
    label = classification_labels[classification]
    
    reader.options['color'][label] = color
    
if include_ruler:
    reader.options['color']['ruler'] = [ 1., 1., 1., 1. ]
if include_disk:
    reader.options['color']['disk'] = [ 1., 1., 1., 1. ]


# In[52]:


## do raw particle data
for part_type in originalReturnParts:
    reader.returnKeys[part_type]=['Coordinates','Velocities']
    reader.addFilter[part_type]=[False,False]
    reader.dolog[part_type]=[False,False]

## set the camera center to be at the origin (defaults to np.mean(Coordinates) otherwise)
##     later on we subtract out halo_center from coordinates but could instead make this halo_center
reader.options['center'] = np.array([0., 0., 0.])

## initialize filter flags and options
reader.defineFilterKeys()
print reader.returnKeys
## load in return keys from snapshot
filenames_opened = reader.populate_dict()


# ## Add Linefinder Data

# In[53]:


# Positions and Velocities
for classification in classification_list:
    
    reader.addtodict(
        reader.partsDict,
        None,
        classification,
        'Coordinates',
         sendlog = 0, 
         sendmag = 0,
        vals = positions[classification],
        filterFlag = False
    )
    reader.addtodict(
        reader.partsDict,
        None,
        classification,
        'Velocities',
        sendlog = 0, 
        sendmag = 0,
        vals = velocities[classification],
        filterFlag = False
    )


# In[54]:


# Filter by Filter Keys

for filter_key in filter_data.keys():
    
    specific_filter_data = filter_data[filter_key]
    
    for classification in classification_list:
        
        if classification not in specific_filter_data.keys():
            continue
            
        log_flag = filter_key in log_filters
        
        reader.addtodict(
            reader.partsDict,
            None,
            classification,
            filter_labels[filter_key],
            sendlog = log_flag, 
            sendmag = 0,
            vals = specific_filter_data[classification],
            filterFlag = True,
        )


# In[55]:


if include_ruler:
    reader.addtodict(
        reader.partsDict,
        None,
        'ruler',
        'Coordinates',
         sendlog = 0, 
         sendmag = 0,
        vals = positions['ruler'],
        filterFlag = False
    )
    reader.addtodict(
        reader.partsDict,
        None,
        'ruler',
        'Velocities',
        sendlog = 0, 
        sendmag = 0,
        vals = np.zeros( positions['ruler'].shape ),
        filterFlag = False
    )


# In[56]:


if include_disk:
    reader.addtodict(
        reader.partsDict,
        None,
        'disk',
        'Coordinates',
         sendlog = 0, 
         sendmag = 0,
        vals = positions['disk'],
        filterFlag = False
    )
    reader.addtodict(
        reader.partsDict,
        None,
        'disk',
        'Velocities',
        sendlog = 0, 
        sendmag = 0,
        vals = np.zeros( positions['disk'].shape ),
        filterFlag = False
    )


# ## Account for Halo Centers

# In[57]:


for part_type in originalReturnParts:
    reader.partsDict[part_type]['Coordinates'] -= w.origin[:,ind]
    reader.partsDict[part_type]['Velocities'] -= w.vel_origin[:,ind]


# ## Write the Files

# In[58]:


## let's shuffle + decimate, add the GUI friendly names
for part_type in originalReturnParts:
    reader.shuffle_dict([part_type]) # Only decimate raw data
reader.swap_dict_names()


# In[63]:


regime_mapping = {
    172: 'highz',
    465: 'lowz',
}
save_base_dir = '{}_{}'.format( sim_name, regime_mapping[snum] )


# In[67]:


# Write the json files
reader.dataDir = os.path.join( target_dir, save_base_dir ) 
reader.cleanDataDir = True
reader.createJSON( overwrite=False )

