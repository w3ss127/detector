import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Input directory (where parquet files are located)
PARQUET_DIR = "huggingface/traini-datasets"
# Output directory for pt files
PT_DIR = "huggingface/traini-datasets"
os.makedirs(PT_DIR, exist_ok=True)

def parquet_to_pt_gpu(parquet_path, pt_path, gpu_id):
    """Convert images from parquet file back to tensor format and save as .pt file using specified GPU"""
    try:
        # Check if file exists and has content
        if not os.path.exists(parquet_path):
            print(f"GPU {gpu_id}: Error: File does not exist: {parquet_path}")
            return
            
        file_size = os.path.getsize(parquet_path)
        if file_size == 0:
            print(f"GPU {gpu_id}: Error: Empty file: {parquet_path}")
            return
            
        # Check if file is actually a parquet file by reading magic numbers
        try:
            with open(parquet_path, 'rb') as f:
                # Check file header (first 4 bytes should be 'PAR1')
                header = f.read(4)
                if header != b'PAR1':
                    print(f"GPU {gpu_id}: Error: Not a valid parquet file (wrong header): {os.path.basename(parquet_path)}")
                    # Try to delete invalid file
                    try:
                        os.remove(parquet_path)
                        print(f"GPU {gpu_id}: Deleted invalid parquet file {os.path.basename(parquet_path)}")
                    except:
                        pass
                    return
                
                # Check file footer (last 4 bytes should also be 'PAR1')
                f.seek(-4, 2)  # Seek to 4 bytes from end
                footer = f.read(4)
                if footer != b'PAR1':
                    print(f"GPU {gpu_id}: Error: Not a valid parquet file (wrong footer): {os.path.basename(parquet_path)}")
                    # Try to delete invalid file
                    try:
                        os.remove(parquet_path)
                        print(f"GPU {gpu_id}: Deleted invalid parquet file {os.path.basename(parquet_path)}")
                    except:
                        pass
                    return
                    
                print(f"GPU {gpu_id}: Valid parquet file detected: {os.path.basename(parquet_path)}")
        except Exception as header_error:
            print(f"GPU {gpu_id}: Error checking file header for {os.path.basename(parquet_path)}: {str(header_error)}")
            return
            
        # Set the device for this process
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Read the parquet file with error handling
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as parquet_error:
            print(f"GPU {gpu_id}: Error reading parquet file {os.path.basename(parquet_path)}: {str(parquet_error)}")
            print(f"GPU {gpu_id}: File size: {file_size} bytes")
            # Try to delete corrupted file
            try:
                os.remove(parquet_path)
                print(f"GPU {gpu_id}: Deleted corrupted parquet file {os.path.basename(parquet_path)}")
            except:
                pass
            return
        
        # Check if 'image' column exists
        if 'image' not in df.columns:
            print(f"GPU {gpu_id}: Error: No 'image' column found in {os.path.basename(parquet_path)}")
            return
            
        # Convert the 'image' column back to tensors
        images_list = df['image'].tolist()
        
        # Convert each image list back to tensor on GPU
        tensors = []
        for i, img_list in enumerate(images_list):
            try:
                # Convert list to tensor and reshape to [3, 256, 256]
                img_tensor = torch.tensor(img_list, dtype=torch.uint8, device=device)
                # Reshape from flat array to [3, 256, 256]
                img_tensor = img_tensor.view(3, 256, 256)
                tensors.append(img_tensor)
            except Exception as img_error:
                print(f"GPU {gpu_id}: Error processing image {i} in {os.path.basename(parquet_path)}: {str(img_error)}")
                continue
        
        # Stack all tensors into a single tensor
        if tensors:
            # Stack tensors to get shape [N, 3, 256, 256] where N is number of images
            stacked_tensor = torch.stack(tensors)
            # Move back to CPU for saving (torch.save works better with CPU tensors)
            stacked_tensor_cpu = stacked_tensor.cpu()
            torch.save(stacked_tensor_cpu, pt_path)
            print(f"GPU {gpu_id}: Saved tensor with shape: {stacked_tensor.shape} from {os.path.basename(parquet_path)}")
            
            # Verification step: reload and check shape/dtype
            try:
                loaded_tensor = torch.load(pt_path, map_location='cpu')
                print(f"GPU {gpu_id}: Verified {os.path.basename(pt_path)}: shape={tuple(loaded_tensor.shape)}, dtype={loaded_tensor.dtype}")
                if loaded_tensor.shape[0] == 0 or loaded_tensor.shape[1:] != (3, 256, 256):
                    print(f"GPU {gpu_id}: WARNING: {os.path.basename(pt_path)} has invalid shape: {tuple(loaded_tensor.shape)}")
            except Exception as verify_error:
                print(f"GPU {gpu_id}: ERROR verifying {os.path.basename(pt_path)}: {str(verify_error)}")
            
            # Delete the parquet file after successful conversion
            try:
                os.remove(parquet_path)
                print(f"GPU {gpu_id}: Deleted parquet file {os.path.basename(parquet_path)}")
            except Exception as delete_error:
                print(f"GPU {gpu_id}: Warning: Could not delete {parquet_path}: {str(delete_error)}")
        else:
            print(f"GPU {gpu_id}: Warning: No valid images found in {os.path.basename(parquet_path)}")
            
    except Exception as e:
        print(f"GPU {gpu_id}: Error processing {os.path.basename(parquet_path)}: {str(e)}")

def process_files_on_gpu(file_list, gpu_id):
    """Process a list of files on a specific GPU"""
    print(f"GPU {gpu_id}: Starting to process {len(file_list)} files")
    for i, parquet_file in enumerate(file_list, 1):
        base = os.path.basename(parquet_file)
        base_noext = os.path.splitext(base)[0]
        pt_file = f"{base_noext}.pt"
        pt_path = os.path.join(PT_DIR, pt_file)
        
        print(f"GPU {gpu_id}: Processing file {i}/{len(file_list)}: {base}")
        parquet_to_pt_gpu(parquet_file, pt_path, gpu_id)
    print(f"GPU {gpu_id}: Completed processing all {len(file_list)} files")

if __name__ == "__main__":
    # Check available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    else:
        num_gpus = 1
        print("No CUDA GPUs available, using CPU")
    
    # Find all parquet files in the huggingface directory
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    print(f"📊 Found {len(parquet_files)} parquet files to convert")
    
    if parquet_files:
        print("\n📋 Files to be converted:")
        for i, file in enumerate(parquet_files[:5], 1):  # Show first 5 files
            print(f"  {i}. {os.path.basename(file)}")
        if len(parquet_files) > 5:
            print(f"  ... and {len(parquet_files) - 5} more files")
    else:
        print("❌ No parquet files found to convert!")
        exit(1)
    
    if num_gpus > 1:
        # Distribute files across GPUs
        files_per_gpu = len(parquet_files) // num_gpus
        remainder = len(parquet_files) % num_gpus
        
        file_distribution = []
        start_idx = 0
        
        for gpu_id in range(num_gpus):
            # Add one extra file to the first 'remainder' GPUs
            extra_file = 1 if gpu_id < remainder else 0
            end_idx = start_idx + files_per_gpu + extra_file
            gpu_files = parquet_files[start_idx:end_idx]
            file_distribution.append((gpu_files, gpu_id))
            start_idx = end_idx
        
        print(f"Distributing files across {num_gpus} GPUs:")
        for gpu_id, (files, _) in enumerate(file_distribution):
            print(f"  GPU {gpu_id}: {len(files)} files")
        
        # Create and start processes for each GPU
        processes = []
        print(f"\n🚀 Starting conversion of {len(parquet_files)} parquet files across {num_gpus} GPUs...")
        for files, gpu_id in file_distribution:
            if files:  # Only create process if there are files to process
                print(f"📁 GPU {gpu_id}: Will process {len(files)} files")
                p = mp.Process(target=process_files_on_gpu, args=(files, gpu_id))
                p.start()
                processes.append(p)
        
        # Wait for all processes to complete
        print(f"\n⏳ Waiting for all {len(processes)} GPU processes to complete...")
        for i, p in enumerate(processes):
            p.join()
            print(f"✅ GPU process {i+1}/{len(processes)} completed")
            
        print("\n🎉 All GPU processes completed successfully!")
        
    else:
        # Single GPU or CPU processing
        print(f"\n🚀 Starting conversion of {len(parquet_files)} parquet files on single device...")
        for i, parquet_file in enumerate(tqdm(parquet_files, desc="Converting parquet to pt"), 1):
            base = os.path.basename(parquet_file)
            base_noext = os.path.splitext(base)[0]
            pt_file = f"{base_noext}.pt"
            pt_path = os.path.join(PT_DIR, pt_file)
            
            print(f"📁 Processing file {i}/{len(parquet_files)}: {base}")
            parquet_to_pt_gpu(parquet_file, pt_path, 0)
        print(f"\n🎉 Completed conversion of all {len(parquet_files)} files!")

    # After all conversions, print a summary of .pt files
    print("\n📦 Summary of .pt files in output directory:")
    pt_files = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))
    if not pt_files:
        print("❌ No .pt files found!")
    else:
        for pt_file in pt_files:
            try:
                tensor = torch.load(pt_file, map_location='cpu')
                print(f"  {os.path.basename(pt_file)}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
            except Exception as e:
                print(f"  {os.path.basename(pt_file)}: ERROR loading file: {str(e)}")