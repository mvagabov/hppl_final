import numpy as np
import cupy as cp
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import display

def load_tiff(file_path):
    return np.array(tiff.imread(file_path))

def memory_stack(stack):
    """
    reshapes stack from (time, height, width) to (height, width, time) 
    for efficient gpu memory access
    
    parameters:
    stack (np.array): 3D numpy array (time, height, width)
    
    returns:
    cp.array: reshaped array (height, width, time)
    """
    return cp.array(np.transpose(stack, (1, 2, 0)))

def show_stack(stack, init_coords=None, start_frame=0, grid_size=None, avg_corr_values=None, marker=5):
    """
    displays 3D image stack with time slider, optional coordinates and grid
    
    parameters:
    stack (np.array): 3D numpy array (time, height, width)
    init_coords (np.array): coordinates to highlight
    start_frame (int): frame number to start display from
    grid_size (int): size of grid cells (None for no grid)
    avg_corr_values (np.array): average cross-correlation values for coloring
    """
    plt.ion()
    
    # slice stack from start_frame
    stack = stack[start_frame:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)
    
    # show first frame
    img = ax.imshow(stack[0], cmap='binary_r')
    
    # plot coordinates if provided
    if init_coords is not None:
        if avg_corr_values is not None:
            # normalize avg_corr_values for coloring
            norm = plt.Normalize(vmin=np.min(avg_corr_values), vmax=np.max(avg_corr_values))
            colors = plt.cm.viridis(norm(avg_corr_values))
            for coord, color in zip(init_coords, colors):
                ax.plot(coord[1], coord[0], 'o', color=color, markersize=marker, alpha=0.7)
        else:
            ax.plot(init_coords[:, 1], init_coords[:, 0], 'r.', markersize=marker, alpha=0.7)
    
    # add grid if size provided
    if grid_size is not None:
        height, width = stack[0].shape
        # add vertical lines
        for x in range(0, width, grid_size):
            ax.axvline(x=x, color='w', alpha=0.3, linestyle='-')
        # add horizontal lines
        for y in range(0, height, grid_size):
            ax.axhline(y=y, color='w', alpha=0.3, linestyle='-')
    
    ax.set_title(f'frame: {start_frame}/{start_frame + len(stack)-1}')
    
    # create slider
    slider_ax = plt.axes([0.1, 0.05, 0.65, 0.03])
    time_slider = Slider(
        ax=slider_ax,
        label='Time',
        valmin=start_frame,
        valmax=start_frame + len(stack)-1,
        valinit=start_frame,
        valstep=1
    )
    
    def update(val):
        frame = int(time_slider.val - start_frame)
        img.set_array(stack[frame])
        ax.set_title(f'frame: {time_slider.val}/{start_frame + len(stack)-1}')
        fig.canvas.draw_idle()
    
    time_slider.on_changed(update)
    plt.show()



def find_peaks(stack, window_size=10):
    """
    finds local maxima in total intensity over time
    
    parameters:
    stack (cp.array): 3D cupy array (height, width, time)
    
    returns:
    tuple: (first_peak_frame, intensities, sum_over_time)
    """
    # sum intensities for each time point
    intensities = cp.sum(stack, axis=(0, 1))
    
    # find local maxima with sliding window
    peaks = []
    for i in range(window_size, len(intensities)-window_size):
        window = intensities[i-window_size:i+window_size+1]
        if cp.all(intensities[i] >= window) and intensities[i] > 0:
            peaks.append(i)
    peaks = cp.array(peaks)
    
    # sum over time
    sum_over_time = cp.sum(stack, axis=2)
    
    return cp.asnumpy(peaks), cp.asnumpy(intensities), cp.asnumpy(sum_over_time)

def plot_intensity_profile(intensities, peaks):
    """
    plots intensity profile with marked peaks
    
    parameters:
    intensities (np.array): 1D array of intensities over time
    peaks (np.array): frame numbers of peaks
    """
    plt.figure(figsize=(10, 4))
    plt.plot(intensities, 'k-', label='intensity')
    plt.plot(peaks, intensities[peaks], 'r*', 
            markersize=3, label='peaks')
    plt.xlabel('frame')
    plt.ylabel('total intensity')
    plt.legend()
    plt.show()

def find_local_maxima(stack, first_peak, grid_size=8):
    """
    finds local maxima in grid cells using CUDA kernel
    
    parameters:
    stack (cp.array): 3D cupy array (height, width, time)
    first_peak (int): frame number to analyze
    grid_size (int): size of grid cells
    
    returns:
    np.array: coordinates of maximum intensity pixels
    """
    height, width = stack.shape[:2]
    peak_frame = stack[:, :, first_peak]
    
    # calculate grid dimensions
    n_cells_h = (height + grid_size - 1) // grid_size
    n_cells_w = (width + grid_size - 1) // grid_size
    
    # prepare output array for maxima coordinates
    max_coords = cp.zeros((n_cells_h * n_cells_w, 2), dtype=cp.int32)
    
    # CUDA kernel for finding local maxima
    kernel_code = r'''
    extern "C" __global__ void find_maxima(
        const float* frame,
        int* max_coords,
        int height,
        int width,
        int grid_size
    ) {
        // get cell index
        int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int n_cells_w = (width + grid_size - 1) / grid_size;
        
        // calculate cell position
        int cell_y = (cell_idx / n_cells_w) * grid_size;
        int cell_x = (cell_idx % n_cells_w) * grid_size;
        
        // check if this thread should process a cell
        if (cell_y >= height || cell_x >= width) return;
        
        // find boundaries of this cell
        int y_end = min(cell_y + grid_size, height);
        int x_end = min(cell_x + grid_size, width);
        
        // find maximum in cell
        float max_val = -1.0f;
        int max_y = cell_y;
        int max_x = cell_x;
        
        for (int y = cell_y; y < y_end; y++) {
            for (int x = cell_x; x < x_end; x++) {
                float val = frame[y * width + x];
                if (val > max_val) {
                    max_val = val;
                    max_y = y;
                    max_x = x;
                }
            }
        }
        
        // store result
        max_coords[cell_idx * 2] = max_y;
        max_coords[cell_idx * 2 + 1] = max_x;
    }
    '''
    
    # compile the kernel
    module = cp.RawModule(code=kernel_code)
    kernel = module.get_function('find_maxima')
    
    # prepare grid and block dimensions
    n_cells = n_cells_h * n_cells_w
    threads_per_block = 1
    blocks = (n_cells + threads_per_block - 1) // threads_per_block
    
    # run kernel
    kernel(
        (blocks,), 
        (threads_per_block,),
        (peak_frame, max_coords, height, width, grid_size)
    )
    
    return cp.asnumpy(max_coords)

def cross_correlate_maxima_kernel(mem_stack, max_coords, top_n=32):
    """
    computes cross-correlation of intensity profiles over time for top N maxima using raw kernels
    
    parameters:
    mem_stack (cp.array): 3D cupy array (height, width, time)
    max_coords (np.array): coordinates of maximum intensity pixels (y, x)
    top_n (int): number of top maxima to consider
    
    returns:
    list of tuples: [(average cross-correlation, coordinates of maxima), ...]
    """
    # ensure top_n does not exceed the number of maxima
    top_n = min(top_n, len(max_coords))
    
    # extract intensity profiles for top N maxima
    y_coords, x_coords = max_coords[:top_n].T
    intensity_profiles = mem_stack[y_coords, x_coords, :].astype(cp.float32)  # ensure float type
    
    # normalize intensity profiles
    intensity_profiles -= cp.mean(intensity_profiles, axis=1, keepdims=True)
    intensity_profiles /= cp.std(intensity_profiles, axis=1, keepdims=True)
    
    # prepare output array for cross-correlation
    cross_corr_matrix = cp.zeros((top_n, top_n), dtype=cp.float32)
    
    # define the raw kernel
    kernel_code = r'''
    extern "C" __global__ void cross_correlate(
        const float* profiles,
        float* cross_corr_matrix,
        int num_profiles,
        int time_length
    ) {
        int i = blockIdx.x;
        int j = threadIdx.x;
        
        if (i < num_profiles && j < num_profiles) {
            float sum = 0.0;
            for (int t = 0; t < time_length; ++t) {
                sum += profiles[i * time_length + t] * profiles[j * time_length + t];
            }
            cross_corr_matrix[i * num_profiles + j] = sum;
        }
    }
    '''
    
    # compile the kernel
    module = cp.RawModule(code=kernel_code)
    kernel = module.get_function('cross_correlate')
    
    # get the number of time points
    time_length = intensity_profiles.shape[1]
    
    # run the kernel
    kernel(
        (top_n,),  # number of blocks
        (top_n,),  # number of threads per block
        (intensity_profiles, cross_corr_matrix, top_n, time_length)
    )
    
    cross_corr_matrix_np = cp.asnumpy(cross_corr_matrix)

    # calculate average excluding the maximum value of correlation with itself
    avg_corr_values = np.mean(cross_corr_matrix_np, axis=1) - np.max(cross_corr_matrix_np, axis=1)

    # pair average cross-correlation with coordinates
    corrcoef_coordinates = [(avg_corr_values[i], (y_coords[i], x_coords[i])) for i in range(top_n)]
    return cross_corr_matrix_np, corrcoef_coordinates

def normalize_corrcoefs(corrcoefs):
    """
    normalizes correlation coefficients to [0, 1] range
    
    parameters:
    corrcoefs (np.array): array of correlation coefficients
    
    returns:
    np.array: normalized correlation coefficients
    """
    # ensure array type
    corrcoefs = np.array(corrcoefs)
    
    # handle case where all values are the same
    if np.max(corrcoefs) == np.min(corrcoefs):
        return np.ones_like(corrcoefs)
    
    # normalize to [0, 1]
    normalized = (corrcoefs - np.min(corrcoefs)) / (np.max(corrcoefs) - np.min(corrcoefs))
    
    return normalized


def plot_intensity_profiles(mem_stack, max_coords, norm_corr_values, peaks):
    """
    plots all intensity profiles colored by their correlation coefficients
    
    parameters:
    mem_stack (cp.array): 3D array (height, width, time)
    max_coords (np.array): coordinates of maxima
    avg_corr_values (np.array): correlation coefficients for each profile
    peaks (np.array): peak frame numbers
    """
    # Extract intensity profiles
    y_coords, x_coords = zip(*max_coords)
    intensity_profiles = cp.asnumpy(mem_stack[y_coords, x_coords, :])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create colormap based on correlation coefficients
    colors = plt.cm.viridis(norm_corr_values)
    
    # Plot each profile
    for profile, color in zip(intensity_profiles, colors):
        plt.plot(profile, color=color, alpha=0.7, linewidth=1)
    
    # Add vertical line for peak frame
    plt.axvline(x=peaks[0], color='r', linestyle='--', alpha=0.5, label='Peak frame')
    
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.title('Intensity Profiles Colored by Correlation')
    plt.legend()
    plt.show()

def adventurer(mem_stack, init_coords, how_far=3, grid_step=4):
    """
    explores and correlates neighboring coordinates for each initial coordinate
    
    parameters:
    mem_stack (cp.array): 3D cupy array (height, width, time)
    init_coords (np.array): initial coordinates to explore from [(y1,x1), (y2,x2), ...]
    how_far (int): how many grid steps to explore in each direction
    grid_step (int): size of each step in pixels (default=4)
    
    returns:
    dict: mapping of initial coordinates to their neighbors' correlation values
        {(y,x): [(corr_val, (neighbor_y, neighbor_x)), ...], ...}
    """
    height, width, _ = mem_stack.shape
    exploration_results = {}
    
    # Create grid offsets
    range_vals = np.arange(-how_far, how_far + 1) * grid_step
    y_offsets, x_offsets = np.meshgrid(range_vals, range_vals)
    offsets = np.column_stack((y_offsets.ravel(), x_offsets.ravel()))
    
    for init_y, init_x in init_coords:
        # generate neighbor coordinates
        neighbor_coords = np.array([(init_y + dy, init_x + dx) for dy, dx in offsets])
        
        # filter out coordinates outside image bounds
        valid_mask = (
            (neighbor_coords[:, 0] >= 0) & 
            (neighbor_coords[:, 0] < height) & 
            (neighbor_coords[:, 1] >= 0) & 
            (neighbor_coords[:, 1] < width)
        )
        valid_neighbors = neighbor_coords[valid_mask]
        
        # extract intensity profiles
        y_coords, x_coords = valid_neighbors.T
        profiles = mem_stack[y_coords, x_coords, :].astype(cp.float32)
        init_profile = mem_stack[init_y, init_x, :].astype(cp.float32)
        
        # normalize profiles
        profiles -= cp.mean(profiles, axis=1, keepdims=True)
        profiles /= cp.std(profiles, axis=1, keepdims=True)
        init_profile -= cp.mean(init_profile)
        init_profile /= cp.std(init_profile)
        
        # prepare output array
        correlations = cp.zeros(len(valid_neighbors), dtype=cp.float32)
        
        # define correlation kernel
        kernel_code = r'''
        extern "C" __global__ void correlate_profiles(
            const float* profiles,
            const float* init_profile,
            float* correlations,
            int num_profiles,
            int time_length
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < num_profiles) {
                float sum = 0.0;
                for (int t = 0; t < time_length; ++t) {
                    sum += profiles[idx * time_length + t] * init_profile[t];
                }
                correlations[idx] = sum;
            }
        }
        '''
        
        # compile and run kernel
        module = cp.RawModule(code=kernel_code)
        kernel = module.get_function('correlate_profiles')
        
        threads_per_block = 256
        blocks = (len(valid_neighbors) + threads_per_block - 1) // threads_per_block
        
        kernel(
            (blocks,),
            (threads_per_block,),
            (profiles, init_profile, correlations, len(valid_neighbors), profiles.shape[1])
        )
        
        corrcoef_coordinates = list(zip(
            cp.asnumpy(correlations),
            valid_neighbors
        ))
        # Normalize correlation values
        corr_values, coords = zip(*corrcoef_coordinates)
        norm_corr_values = cp.asnumpy(normalize_corrcoefs(cp.array(corr_values)))
        
        # Store results
        exploration_results[(init_y, init_x)] = list(zip(norm_corr_values, coords))
    
    return exploration_results
