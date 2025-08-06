import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTModel
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import gc
from tqdm import tqdm
import json
from datetime import datetime
import time
import signal
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optimize for multi-GPU testing speed
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"Found {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU {i}: {gpu_memory:.1f} GB")
    print("Optimized for multi-GPU testing speed")

# Memory management
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def setup_distributed(rank, world_size, master_port='29500'):
    """Setup distributed testing environment with retry mechanism"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TIMEOUT'] = '1800'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_SOCKET_NTHREADS'] = '4'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    
    max_retries = 5
    base_port = int(master_port)
    
    for attempt in range(max_retries):
        try:
            current_port = str(base_port + attempt)
            os.environ['MASTER_PORT'] = current_port
            
            print(f"Rank {rank}: Attempting to connect on port {current_port} (attempt {attempt + 1}/{max_retries})")
            
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            
            print(f"Rank {rank}: Distributed testing initialized successfully on port {current_port}")
            return
            
        except Exception as e:
            print(f"Rank {rank}: Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Rank {rank}: Retrying with different port...")
                time.sleep(2)
            else:
                print(f"Rank {rank}: All retry attempts failed")
                raise e

def cleanup_distributed():
    """Cleanup distributed testing environment"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed testing cleaned up")

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup_distributed()
    exit(0)

def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available(): 
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        memory_info[f'gpu_{i}'] = {
            'total': torch.cuda.get_device_properties(i).total_memory / 1e9,
            'allocated': torch.cuda.memory_allocated(i) / 1e9,
            'cached': torch.cuda.memory_reserved(i) / 1e9,
            'free': (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1e9
        }
    return memory_info

def verify_multi_gpu_usage(rank, world_size):
    """Verify that multiple GPUs are being used correctly"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("MULTI-GPU USAGE VERIFICATION")
        print('='*60)
        
        if world_size > 1:
            if dist.is_initialized():
                print(f"✅ Distributed training initialized with {world_size} processes")
                print(f"✅ Backend: {dist.get_backend()}")
                print(f"✅ Rank: {dist.get_rank()}")
                print(f"✅ World Size: {dist.get_world_size()}")
            else:
                print("❌ Distributed training not initialized")
                return False
        
        memory_info = get_gpu_memory_info()
        print(f"\nGPU Memory Usage:")
        total_allocated = 0
        for gpu_id, info in memory_info.items():
            print(f"  {gpu_id.upper()}: {info['allocated']:.1f}GB / {info['total']:.1f}GB "
                  f"({info['allocated']/info['total']*100:.1f}%)")
            total_allocated += info['allocated']
        
        if world_size > 1:
            print(f"\nTotal GPU Memory Allocated: {total_allocated:.1f}GB")
            if total_allocated > 1.0:
                print("✅ Multiple GPUs are being used (significant memory allocation detected)")
                return True
            else:
                print("⚠️  Low GPU memory usage - may not be using multiple GPUs effectively")
                return False
        else:
            print("ℹ️  Single GPU mode - no multi-GPU verification needed")
            return True

class FloatNormalize:
    def __call__(self, x):
        return x.float() / 255.0

class PtFileTestDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_name=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_name = class_name
        self.pt_files = []
        self.labels = []
        self.max_samples = max_samples
        
        self.class_to_idx = {'real': 1, 'synthetic': 0, 'semi-synthetic': -1}
        
        if class_name:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for pt_file in glob(os.path.join(class_dir, "*.pt")):
                    self.pt_files.append(pt_file)
                    self.labels.append(self.class_to_idx.get(class_name, -1))
        else:
            for class_name in os.listdir(root_dir):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for pt_file in glob(os.path.join(class_dir, "*.pt")):
                        self.pt_files.append(pt_file)
                        self.labels.append(self.class_to_idx.get(class_name, -1))
        
        self.images_per_file = 5000
        
        if self.max_samples:
            total_samples = len(self.pt_files) * self.images_per_file
            if total_samples > self.max_samples:
                files_needed = max(1, self.max_samples // self.images_per_file)
                self.pt_files = self.pt_files[:files_needed]
                self.labels = self.labels[:files_needed]
                print(f"Limited to {files_needed} files ({self.max_samples} samples) for faster testing")
        
        print(f"Loaded {len(self.pt_files)} .pt files from {root_dir}")

    def __len__(self):
        return len(self.pt_files) * self.images_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.images_per_file
        img_idx = idx % self.images_per_file
        pt_file = self.pt_files[file_idx]
        label = self.labels[file_idx]

        data = torch.load(pt_file, map_location='cpu')
        img = data[img_idx]

        if self.transform:
            img = self.transform(img)

        return img, label, pt_file

class ConvNeXtViTAttention(nn.Module):
    def __init__(self):
        super(ConvNeXtViTAttention, self).__init__()
        self.convnext = torch.hub.load('pytorch/vision', 'convnext_base', weights='IMAGENET1K_V1')
        self.convnext.classifier[2] = nn.Identity()
        self.convnext_features = 1024

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_features = 768

        self.vit_projection = nn.Linear(self.vit_features, self.convnext_features)
        self.attention = nn.MultiheadAttention(embed_dim=self.convnext_features, num_heads=8)
        self.fc1 = nn.Linear(self.convnext_features + self.convnext_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        convnext_out = self.convnext(x)
        convnext_out = convnext_out.unsqueeze(0)

        vit_out = self.vit(pixel_values=x).last_hidden_state[:, 0, :]
        vit_out = self.vit_projection(vit_out)
        vit_out = vit_out.unsqueeze(0)

        attn_output, _ = self.attention(convnext_out, convnext_out, vit_out)
        attn_output = attn_output.squeeze(0)

        combined = torch.cat((convnext_out.squeeze(0), vit_out.squeeze(0)), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def evaluate_model(model, data_loader, class_name, threshold=0.5, rank=0, world_size=1, semi_synthetic_min=0.4, semi_synthetic_max=0.6):
    """Evaluate model on a dataset and return metrics (optimized for multi-GPU speed)"""
    model.eval()
    all_outputs = []
    all_labels = []
    all_predictions = []
    
    if rank == 0:
        print(f"Evaluating on {class_name} data...")
        print(f"Device: {device}, World Size: {world_size}")
    
    batch_outputs = []
    batch_labels = []
    batch_predictions = []
    
    if rank == 0 and world_size > 1:
        print(f"GPU Memory before evaluation:")
        memory_info = get_gpu_memory_info()
        for gpu_id, info in memory_info.items():
            print(f"  {gpu_id.upper()}: {info['allocated']:.1f}GB / {info['total']:.1f}GB")
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(data_loader, desc=f'Evaluating {class_name}', disable=rank != 0)):
            try:
                images = images.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images).squeeze()
                
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                
                batch_size = outputs.size(0)
                predictions = torch.zeros(batch_size, 3, device=device)
                
                real_mask = outputs >= semi_synthetic_max
                synthetic_mask = outputs <= semi_synthetic_min
                semi_synthetic_mask = (outputs > semi_synthetic_min) & (outputs < semi_synthetic_max)
                
                predictions[real_mask, 0] = 1
                predictions[synthetic_mask, 1] = 1
                predictions[semi_synthetic_mask, 2] = 1
                
                batch_outputs.append(outputs.cpu())
                batch_labels.append(labels.cpu())
                batch_predictions.append(predictions.cpu())
                
                if batch_idx % 10 == 0:
                    del images
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"GPU OOM in batch {batch_idx}. Skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    if rank == 0:
                        print(f"Runtime error in batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                if rank == 0:
                    print(f"Unexpected error in batch {batch_idx}: {e}")
                continue
    
    if len(batch_outputs) > 0:
        all_outputs = torch.cat(batch_outputs, dim=0).numpy()
        all_labels = torch.cat(batch_labels, dim=0).numpy()
        all_predictions = torch.cat(batch_predictions, dim=0).numpy()
    else:
        all_outputs = np.array([])
        all_labels = np.array([])
        all_predictions = np.array([[]]).reshape(0, 3)
    
    if world_size > 1 and len(all_outputs) > 0:
        outputs_tensor = torch.from_numpy(all_outputs).to(device)
        labels_tensor = torch.from_numpy(all_labels).to(device)
        predictions_tensor = torch.from_numpy(all_predictions).to(device)
        
        gathered_outputs = [torch.zeros_like(outputs_tensor) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
        gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_outputs, outputs_tensor)
        dist.all_gather(gathered_labels, labels_tensor)
        dist.all_gather(gathered_predictions, predictions_tensor)
        
        all_outputs = torch.cat(gathered_outputs, dim=0).cpu().numpy()
        all_labels = torch.cat(gathered_labels, dim=0).cpu().numpy()
        all_predictions = torch.cat(gathered_predictions, dim=0).cpu().numpy()
    
    if rank == 0 and world_size > 1:
        print(f"\nGPU Memory after evaluation:")
        memory_info = get_gpu_memory_info()
        for gpu_id, info in memory_info.items():
            print(f"  {gpu_id.upper()}: {info['allocated']:.1f}GB / {info['total']:.1f}GB")
    
    return all_outputs, all_labels, all_predictions

def calculate_metrics(outputs, labels, predictions, class_name, semi_synthetic_min=0.4, semi_synthetic_max=0.6):
    """Calculate comprehensive metrics for three-class predictions"""
    metrics = {}
    
    pred_classes = np.argmax(predictions, axis=1)
    
    if class_name in ['real', 'synthetic']:
        valid_indices = (labels >= 0) & (labels <= 1)
        valid_labels = labels[valid_indices]
        valid_pred_classes = pred_classes[valid_indices]
        
        valid_labels_mapped = np.where(valid_labels == 1, 0, 1)
        
        if len(valid_labels) > 0:
            metrics['accuracy'] = accuracy_score(valid_labels_mapped, valid_pred_classes)
            metrics['precision'] = precision_score(valid_labels_mapped, valid_pred_classes, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(valid_labels_mapped, valid_pred_classes, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(valid_labels_mapped, valid_pred_classes, average='weighted', zero_division=0)
            metrics['mcc'] = matthews_corrcoef(valid_labels_mapped, valid_pred_classes)
            
            cm = confusion_matrix(valid_labels_mapped, valid_pred_classes, labels=[0, 1, 2])
            metrics['confusion_matrix'] = cm.tolist()
            
            metrics['classification_report'] = classification_report(
                valid_labels_mapped, valid_pred_classes, 
                target_names=['Real', 'Synthetic', 'Semi-Synthetic'], 
                zero_division=0, output_dict=True
            )
        else:
            print(f"Warning: No valid labels found for {class_name}")
    
    metrics['mean_output'] = float(np.mean(outputs))
    metrics['std_output'] = float(np.std(outputs))
    metrics['min_output'] = float(np.min(outputs))
    metrics['max_output'] = float(np.max(outputs))
    metrics['median_output'] = float(np.median(outputs))
    
    metrics['outputs_real'] = int(np.sum(pred_classes == 0))
    metrics['outputs_synthetic'] = int(np.sum(pred_classes == 1))
    metrics['outputs_semi_synthetic'] = int(np.sum(pred_classes == 2))
    metrics['total_samples'] = len(outputs)
    
    return metrics

def plot_output_distribution(outputs, class_name, save_dir='binary_test_results', semi_synthetic_min=0.4, semi_synthetic_max=0.6):
    """Plot and save output distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(outputs, bins=50, range=(0, 1), density=True, alpha=0.7, edgecolor='black')
    plt.axvline(x=semi_synthetic_min, color='red', linestyle='--', linewidth=2, label=f'Semi-Synthetic Min ({semi_synthetic_min})')
    plt.axvline(x=semi_synthetic_max, color='red', linestyle='--', linewidth=2, label=f'Semi-Synthetic Max ({semi_synthetic_max})')
    plt.axvline(x=np.mean(outputs), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(outputs):.3f})')
    plt.title(f'Model Output Distribution - {class_name.title()} Images')
    plt.xlabel('Model Output')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{class_name}_output_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, class_name, save_dir='binary_test_results'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Synthetic', 'Semi-Synthetic'], 
                yticklabels=['Real', 'Synthetic', 'Semi-Synthetic'])
    plt.title(f'Confusion Matrix - {class_name.title()} Images')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{class_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def test_worker(rank, world_size, test_dir, model_path, batch_size, max_samples_per_class, save_dir, semi_synthetic_min=0.4, semi_synthetic_max=0.6):
    """Distributed testing worker function"""
    try:
        if world_size > 1:
            setup_distributed(rank, world_size)
        
        device = torch.device(f'cuda:{rank}' if world_size > 1 else 'cuda:0')
        
        if rank == 0:
            print("Loading trained model...")
        
        model = ConvNeXtViTAttention().to(device)
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if rank == 0:
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                    if 'val_mcc' in checkpoint:
                        print(f"Best validation MCC: {checkpoint['val_mcc']:.4f}")
            else:
                model.load_state_dict(checkpoint)
                if rank == 0:
                    print("Loaded model state dict directly")
        except FileNotFoundError:
            if rank == 0:
                print(f"❌ Error: Model file '{model_path}' not found!")
                print("Please ensure the model checkpoint exists at the specified path.")
                print("Available checkpoints in binary_checkpoints/ directory:")
                try:
                    checkpoints = glob.glob('binary_checkpoints/*.pth')
                    for cp in checkpoints:
                        print(f"  - {cp}")
                except:
                    print("  No checkpoints found")
            raise
        except RuntimeError as e:
            if rank == 0:
                print(f"❌ Error: Failed to load model from '{model_path}': {e}")
                print("The checkpoint file may be corrupted or in wrong format.")
                print("Please check if the file exists and is a valid PyTorch checkpoint.")
            raise
        except Exception as e:
            if rank == 0:
                print(f"❌ Error: Unexpected error loading model: {e}")
            raise
        
        if world_size > 1:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        model.eval()
        
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            if rank == 0:
                print("Enabled mixed precision for faster inference")
        
        if rank == 0:
            print("Model loaded successfully!")
        
        verify_multi_gpu_usage(rank, world_size)
        
        test_transform = transforms.Compose([
            FloatNormalize(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_classes = ['real', 'synthetic', 'semi-synthetic']
        all_results = {}
        
        total_start_time = time.time()
        
        for class_name in test_classes:
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.exists(class_dir):
                if rank == 0:
                    print(f"Warning: Directory '{class_dir}' does not exist. Skipping {class_name}.")
                continue
            
            if rank == 0:
                print(f"\n{'='*50}")
                print(f"Testing on {class_name.upper()} images")
                print('='*50)
            
            class_start_time = time.time()
            
            dataset = PtFileTestDataset(test_dir, transform=test_transform, class_name=class_name, max_samples=max_samples_per_class)
            
            if len(dataset) == 0:
                if rank == 0:
                    print(f"No data found for {class_name}. Skipping.")
                continue
            
            if world_size > 1:
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
            else:
                sampler = None
            
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=None
            )
            
            outputs, labels, predictions = evaluate_model(
                model, dataloader, class_name, rank=rank, world_size=world_size,
                semi_synthetic_min=semi_synthetic_min, semi_synthetic_max=semi_synthetic_max
            )
            
            if rank == 0:
                metrics = calculate_metrics(
                    outputs, labels, predictions, class_name,
                    semi_synthetic_min=semi_synthetic_min, semi_synthetic_max=semi_synthetic_max
                )
                all_results[class_name] = metrics
                
                print(f"\nResults for {class_name.upper()}:")
                print(f"Total samples: {metrics['total_samples']}")
                print(f"Mean output: {metrics['mean_output']:.4f}")
                print(f"Std output: {metrics['std_output']:.4f}")
                print(f"Predicted Real [1,0,0]: {metrics['outputs_real']}")
                print(f"Predicted Synthetic [0,1,0]: {metrics['outputs_synthetic']}")
                print(f"Predicted Semi-Synthetic [0,0,1]: {metrics['outputs_semi_synthetic']}")
                
                if class_name in ['real', 'synthetic']:
                    print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
                    print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
                    print(f"F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
                    print(f"MCC: {metrics.get('mcc', 'N/A'):.4f}")
                    if 'classification_report' in metrics:
                        print("\nClassification Report:")
                        for cls, report in metrics['classification_report'].items():
                            if cls in ['Real', 'Synthetic', 'Semi-Synthetic']:
                                print(f"  {cls}:")
                                print(f"    Precision: {report['precision']:.4f}")
                                print(f"    Recall: {report['recall']:.4f}")
                                print(f"    F1-Score: {report['f1-score']:.4f}")
                                print(f"    Support: {report['support']}")
                
                plot_output_distribution(
                    outputs, class_name, save_dir,
                    semi_synthetic_min=semi_synthetic_min, semi_synthetic_max=semi_synthetic_max
                )
                
                if class_name in ['real', 'synthetic'] and 'confusion_matrix' in metrics:
                    plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_name, save_dir)
                
                print(f"Plots saved in '{save_dir}' directory")
                
                class_time = time.time() - class_start_time
                print(f"Time for {class_name}: {class_time:.2f} seconds")
                if metrics['total_samples'] > 0:
                    samples_per_second = metrics['total_samples'] / class_time
                    print(f"Speed: {samples_per_second:.1f} samples/second")
        
        if rank == 0:
            total_time = time.time() - total_start_time
            print(f"\nTotal testing time: {total_time:.2f} seconds")
            
            results_file = os.path.join(save_dir, 'test_results.json')
            all_results['timestamp'] = datetime.now().isoformat()
            all_results['model_path'] = model_path
            all_results['test_directory'] = test_dir
            all_results['world_size'] = world_size
            all_results['semi_synthetic_min'] = semi_synthetic_min
            all_results['semi_synthetic_max'] = semi_synthetic_max
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\n{'='*50}")
            print("TESTING COMPLETED")
            print('='*50)
            print(f"Results saved to: {results_file}")
            print(f"Visualizations saved in: {save_dir}")
            
            print("\nSUMMARY:")
            for class_name, results in all_results.items():
                if class_name in ['timestamp', 'model_path', 'test_directory', 'world_size', 'semi_synthetic_min', 'semi_synthetic_max']:
                    continue
                print(f"{class_name.upper()}: {results['total_samples']} samples, "
                      f"Mean output: {results['mean_output']:.4f}")
                if 'mcc' in results:
                    print(f"  MCC: {results['mcc']:.4f}")
                print(f"  Real [1,0,0]: {results['outputs_real']}")
                print(f"  Synthetic [0,1,0]: {results['outputs_synthetic']}")
                print(f"  Semi-Synthetic [0,0,1]: {results['outputs_semi_synthetic']}")
        
        if world_size > 1:
            cleanup_distributed()
        
    except Exception as e:
        print(f"Rank {rank}: Error in testing: {e}")
        if world_size > 1:
            cleanup_distributed()
        raise

def find_available_checkpoints():
    """Find available checkpoint files"""
    checkpoint_dir = 'binary_checkpoints'
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    return sorted(checkpoints)

def main():
    test_dir = 'datasets/test'
    model_path = 'binary_checkpoints/checkpoint_latest.pth'
    semi_synthetic_min = 0.4
    semi_synthetic_max = 0.6
    
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 20:
            batch_size = 32
        elif gpu_memory_gb >= 10:
            batch_size = 16
        else:
            batch_size = 8
    else:
        world_size = 1
        batch_size = 8
    
    max_samples_per_class = 10000
    save_dir = 'binary_test_results'
    
    print(f"Using {world_size} GPU(s)")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Total batch size: {batch_size * world_size}")
    print(f"Max samples per class: {max_samples_per_class}")
    print(f"Semi-synthetic range: [{semi_synthetic_min}, {semi_synthetic_max}]")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist!")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' does not exist!")
        available_checkpoints = find_available_checkpoints()
        if available_checkpoints:
            print("Found the following checkpoint files:")
            for cp in available_checkpoints:
                print(f"  - {cp}")
            print(f"\nTo use a different checkpoint, modify the 'model_path' variable in the code.")
            print(f"Example: model_path = '{available_checkpoints[0]}'")
        else:
            print("No checkpoint files found in 'binary_checkpoints/' directory.")
            print("Please ensure you have trained the model first using Method_One.py")
        return
    
    if world_size > 1:
        print(f"Starting distributed testing with {world_size} GPUs...")
        try:
            mp.spawn(
                test_worker,
                args=(world_size, test_dir, model_path, batch_size, max_samples_per_class, save_dir, semi_synthetic_min, semi_synthetic_max),
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            print(f"Distributed testing failed: {e}")
            print("Falling back to single GPU testing...")
            test_worker(0, 1, test_dir, model_path, batch_size, max_samples_per_class, save_dir, semi_synthetic_min, semi_synthetic_max)
    else:
        print("Starting single GPU testing...")
        test_worker(0, 1, test_dir, model_path, batch_size, max_samples_per_class, save_dir, semi_synthetic_min, semi_synthetic_max)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    mp.set_start_method('spawn', force=True)
    main()