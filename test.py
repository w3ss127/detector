import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from pathlib import Path
import os
import random
import math
import time
import json
from datetime import datetime
import gc
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# TENSOR BATCH GENERATOR WITH UINT8 SUPPORT
# ============================================================================

class TensorBatchGenerator:
    """
    Generates images and saves them directly as tensor batches (.pt files)
    Each .pt file contains exactly 5000 image tensors with uint8 pixel type
    """
    
    def __init__(self, tensor_size=(224, 224), images_per_batch=5000):
        """
        Initialize the tensor batch generator
        
        Args:
            tensor_size: Target size for images (width, height)
            images_per_batch: Number of images per .pt file (default: 5000)
        """
        self.tensor_size = tensor_size
        self.images_per_batch = images_per_batch
        
        # Transform to convert PIL to tensor with uint8
        self.transform = transforms.Compose([
            transforms.Resize(tensor_size),
            transforms.ToTensor(),  # Converts to [0,1] float32 and (C,H,W) format
        ])
        
        logger.info(f"Initialized TensorBatchGenerator:")
        logger.info(f"  - Target size: {tensor_size}")
        logger.info(f"  - Images per batch: {images_per_batch}")
        logger.info(f"  - Output format: uint8 tensors")
    
    def pil_to_uint8_tensor(self, pil_image):
        """
        Convert PIL Image to uint8 tensor
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            torch.Tensor: uint8 tensor with shape (C, H, W)
        """
        try:
            # Apply transform (resize and convert to float tensor [0,1])
            float_tensor = self.transform(pil_image)
            
            # Convert to uint8 [0,255] range
            uint8_tensor = (float_tensor * 255).clamp(0, 255).byte()
            
            return uint8_tensor
            
        except Exception as e:
            logger.error(f"Error converting PIL to tensor: {e}")
            return None
    
    def save_tensor_batch(self, tensor_list, output_path, batch_metadata=None):
        """
        Save a list of tensors as a single .pt file (TENSOR ONLY)
        
        Args:
            tensor_list: List of uint8 tensors
            output_path: Path to save the .pt file
            batch_metadata: Optional metadata dictionary (saved separately)
            
        Returns:
            bool: Success status
        """
        try:
            if not tensor_list:
                logger.warning("Empty tensor list provided")
                return False
            
            # Stack tensors into batch (N, C, H, W)
            batch_tensor = torch.stack(tensor_list)
            
            # Verify uint8 format
            if batch_tensor.dtype != torch.uint8:
                logger.warning(f"Converting tensor from {batch_tensor.dtype} to uint8")
                batch_tensor = batch_tensor.byte()
            
            # Save ONLY the tensor to .pt file
            torch.save(batch_tensor, output_path)
            
            # Save metadata separately if provided
            if batch_metadata is not None:
                batch_metadata.update({
                    'batch_size': len(tensor_list),
                    'tensor_shape': list(batch_tensor.shape),
                    'dtype': str(batch_tensor.dtype),
                    'pixel_format': 'uint8',
                    'value_range': '[0, 255]',
                    'creation_time': datetime.now().isoformat(),
                    'image_size': self.tensor_size
                })
                
                # Save metadata as JSON file
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(batch_metadata, f, indent=2)
            
            # Log file info
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved tensor batch: {output_path.name}")
            logger.info(f"  - Shape: {batch_tensor.shape}")
            logger.info(f"  - Dtype: {batch_tensor.dtype}")
            logger.info(f"  - Size: {file_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving tensor batch {output_path}: {e}")
            return False

# ============================================================================
# SEMI-SYNTHETIC GENERATOR WITH DIRECT TENSOR OUTPUT
# ============================================================================

class SemiSyntheticTensorGenerator:
    """
    Generates semi-synthetic images and saves them directly as tensor batches
    """
    
    def __init__(self, tensor_size=(224, 224)):
        self.tensor_size = tensor_size
        self.batch_generator = TensorBatchGenerator(tensor_size)
        self.fonts = self._load_fonts()
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (255, 192, 203), (165, 42, 42), (0, 128, 128), (128, 128, 0),
            (255, 165, 0), (128, 128, 128), (0, 0, 128), (128, 0, 0)
        ]
        self.text_samples = {
            'trending': ['TRENDING', 'VIRAL', 'HOT', 'NEW', 'EXCLUSIVE', 'BREAKING'],
            'social': ['LIKE', 'FOLLOW', 'SHARE', 'SUBSCRIBE', 'COMMENT', 'SAVE'],
            'emotions': ['WOW', 'AMAZING', 'SHOCKING', 'INCREDIBLE', 'MIND BLOWN'],
            'generic': ['2024', '2025', 'HD', '4K', 'TOP', 'BEST', 'LIVE'],
            'promo': ['SALE', 'OFFER', 'LIMITED', 'EXCLUSIVE', 'PREMIUM', 'VIP',
                     'FREE', 'BONUS', 'SPECIAL', 'URGENT', 'NOW', 'TODAY']
        }
        
        # Generation styles
        self.styles = ['youtube', 'social', 'trending', 'news', 'advertisement']
        
    def _load_fonts(self):
        """Load fonts with better error handling"""
        fonts = {}
        sizes = [16, 20, 24, 28, 32, 40, 48, 56, 64]
        
        # Try to load system fonts first
        system_fonts = [
            '/System/Library/Fonts/Arial.ttf',  # macOS
            '/Windows/Fonts/arial.ttf',         # Windows
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            '/System/Library/Fonts/Helvetica.ttc',  # macOS alternative
            '/Windows/Fonts/calibri.ttf',       # Windows alternative
        ]
        
        font_path = None
        for path in system_fonts:
            if os.path.exists(path):
                font_path = path
                break
        
        for size in sizes:
            try:
                if font_path:
                    fonts[size] = ImageFont.truetype(font_path, size)
                else:
                    fonts[size] = ImageFont.load_default()
            except Exception as e:
                logger.warning(f"Could not load font size {size}: {e}")
                fonts[size] = ImageFont.load_default()
        
        return fonts

    def _load_and_prepare_image(self, image_path, max_size=(800, 600)):
        """Load and prepare base image"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                return img.copy()
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            return None

    def _add_text_with_outline(self, draw, text, position, font, text_color, outline_color, outline_width=2):
        """Add text with outline for better visibility"""
        x, y = position
        
        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
        
        # Draw main text
        draw.text((x, y), text, fill=text_color, font=font)

    def create_youtube_style(self, base_image_path):
        """Generate YouTube thumbnail style image and return as uint8 tensor"""
        base_img = self._load_and_prepare_image(base_image_path)
        if base_img is None:
            return None
            
        try:
            width, height = base_img.size
            draw = ImageDraw.Draw(base_img)
            
            # Add text overlay
            text = random.choice(self.text_samples['emotions'])
            font_size = min(48, width // 10, height // 8)
            font = self.fonts.get(font_size) or self.fonts[24]
            
            # Calculate text position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = max(20, min(width - text_width - 20, 50))
            y = max(20, min(height - text_height - 20, 50))
            
            # Add text with outline
            self._add_text_with_outline(draw, text, (x, y), font, (255, 255, 255), (0, 0, 0), 3)
            
            # Add play button
            play_size = min(width, height) // 8
            play_x = width // 2 - play_size // 2
            play_y = height // 2 - play_size // 2
            
            # Red circle
            draw.ellipse([play_x, play_y, play_x + play_size, play_y + play_size], 
                        fill=(255, 0, 0))
            
            # White triangle
            triangle_size = play_size // 3
            triangle_x = play_x + play_size // 2 - triangle_size // 2
            triangle_y = play_y + play_size // 2 - triangle_size // 2
            
            triangle_points = [
                (triangle_x, triangle_y),
                (triangle_x, triangle_y + triangle_size),
                (triangle_x + triangle_size, triangle_y + triangle_size // 2)
            ]
            draw.polygon(triangle_points, fill=(255, 255, 255))
            
            # Convert to uint8 tensor
            return self.batch_generator.pil_to_uint8_tensor(base_img)
            
        except Exception as e:
            logger.error(f"Error creating YouTube style: {e}")
            return None

    def create_social_media_style(self, base_image_path):
        """Generate social media style image and return as uint8 tensor"""
        base_img = self._load_and_prepare_image(base_image_path)
        if base_img is None:
            return None
            
        try:
            width, height = base_img.size
            draw = ImageDraw.Draw(base_img)
            
            # Add social media text
            text = random.choice(self.text_samples['social'])
            color = random.choice(self.colors)
            
            font_size = min(32, width // 12, height // 10)
            font = self.fonts.get(font_size) or self.fonts[24]
            
            # Position text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = random.randint(20, max(21, width - text_width - 20))
            y = random.randint(20, max(21, height - text_height - 40))
            
            # Background rectangle
            padding = 15
            bg_bbox = [x - padding, y - padding//2, x + text_width + padding, y + text_height + padding//2]
            draw.rectangle(bg_bbox, fill=color)
            
            # Text
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            # Add hashtag
            hashtag = f"#{random.choice(['viral', 'trending', 'amazing', 'wow', 'epic'])}"
            hashtag_font = self.fonts.get(20) or self.fonts[16]
            hashtag_y = y + text_height + 10
            if hashtag_y < height - 30:
                draw.text((x, hashtag_y), hashtag, fill=color, font=hashtag_font)
            
            return self.batch_generator.pil_to_uint8_tensor(base_img)
            
        except Exception as e:
            logger.error(f"Error creating social media style: {e}")
            return None

    def create_trending_style(self, base_image_path):
        """Generate trending style image and return as uint8 tensor"""
        base_img = self._load_and_prepare_image(base_image_path)
        if base_img is None:
            return None
            
        try:
            width, height = base_img.size
            draw = ImageDraw.Draw(base_img)
            
            # Add trending elements
            text = random.choice(self.text_samples['trending'])
            
            font_size = min(40, width // 8, height // 6)
            font = self.fonts.get(font_size) or self.fonts[32]
            
            # Center the text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = max(20, (width - text_width) // 2)
            y = max(20, height // 4)
            
            # Gradient-like background effect
            bg_color = random.choice(self.colors)
            padding = 25
            
            # Multiple rectangles for gradient effect
            for i in range(5):
                alpha_color = tuple(int(c * (1 - i * 0.1)) for c in bg_color)
                draw.rectangle([
                    x - padding + i*2, 
                    y - padding//2 + i*2, 
                    x + text_width + padding - i*2, 
                    y + text_height + padding//2 - i*2
                ], fill=alpha_color)
            
            # White text with shadow
            self._add_text_with_outline(draw, text, (x, y), font, (255, 255, 255), (0, 0, 0), 2)
            
            # Add decorative elements
            for _ in range(random.randint(8, 15)):
                elem_x = random.randint(0, width - 20)
                elem_y = random.randint(0, height - 20)
                size = random.randint(8, 25)
                color = random.choice(self.colors)
                
                if random.choice([True, False]):
                    # Circle
                    draw.ellipse([elem_x, elem_y, elem_x + size, elem_y + size], fill=color)
                else:
                    # Star-like shape
                    points = []
                    center_x, center_y = elem_x + size//2, elem_y + size//2
                    for i in range(5):
                        angle = i * 2 * math.pi / 5
                        px = center_x + size//2 * math.cos(angle)
                        py = center_y + size//2 * math.sin(angle)
                        points.append((px, py))
                    if len(points) >= 3:
                        draw.polygon(points, fill=color)
            
            return self.batch_generator.pil_to_uint8_tensor(base_img)
            
        except Exception as e:
            logger.error(f"Error creating trending style: {e}")
            return None

    def create_news_style(self, base_image_path):
        """Generate news style image and return as uint8 tensor"""
        base_img = self._load_and_prepare_image(base_image_path)
        if base_img is None:
            return None
            
        try:
            width, height = base_img.size
            draw = ImageDraw.Draw(base_img)
            
            # Add news banner at bottom
            banner_height = max(50, height // 10)
            banner_color = (200, 0, 0)  # News red
            draw.rectangle([0, height - banner_height, width, height], fill=banner_color)
            
            # Add news text
            font_size = min(24, banner_height // 2)
            font = self.fonts.get(font_size) or self.fonts[20]
            
            news_text = random.choice(['BREAKING NEWS', 'LIVE', 'URGENT', 'DEVELOPING', 'EXCLUSIVE'])
            text_y = height - banner_height + (banner_height - font_size) // 2
            
            # White text on red background
            draw.text((20, text_y), news_text, fill=(255, 255, 255), font=font)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M")
            timestamp_bbox = draw.textbbox((0, 0), timestamp, font=font)
            timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
            timestamp_x = width - timestamp_width - 20
            draw.text((timestamp_x, text_y), timestamp, fill=(255, 255, 255), font=font)
            
            return self.batch_generator.pil_to_uint8_tensor(base_img)
            
        except Exception as e:
            logger.error(f"Error creating news style: {e}")
            return None

    def create_advertisement_style(self, base_image_path):
        """Generate advertisement style image and return as uint8 tensor"""
        base_img = self._load_and_prepare_image(base_image_path)
        if base_img is None:
            return None
            
        try:
            width, height = base_img.size
            draw = ImageDraw.Draw(base_img)
            
            # Add promotional text
            promo_texts = ['50% OFF', 'SALE', 'LIMITED TIME', 'BUY NOW', 'SPECIAL OFFER', 'FREE SHIPPING']
            text = random.choice(promo_texts)
            
            font_size = min(48, width // 6, height // 8)
            font = self.fonts.get(font_size) or self.fonts[32]
            
            # Position in corner
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if random.choice([True, False]):
                x = width - text_width - 50  # Top-right
            else:
                x = 30  # Top-left
            
            y = 30
            
            # Star burst background
            bg_size = max(text_width, text_height) + 40
            center_x = x + text_width // 2
            center_y = y + text_height // 2
            
            points = []
            num_points = 8
            outer_radius = bg_size // 2
            inner_radius = outer_radius // 2
            
            for i in range(num_points * 2):
                angle = i * math.pi / num_points
                if i % 2 == 0:
                    radius = outer_radius
                else:
                    radius = inner_radius
                px = center_x + radius * math.cos(angle)
                py = center_y + radius * math.sin(angle)
                points.append((px, py))
            
            if len(points) >= 6:
                draw.polygon(points, fill=(255, 255, 0))  # Yellow burst
                draw.polygon(points, outline=(255, 0, 0), width=3)  # Red outline
            
            # Promotional text
            draw.text((x, y), text, fill=(255, 0, 0), font=font)
            
            # Add badge
            badge_font = self.fonts.get(16) or self.fonts[16]
            badge_text = random.choice(['NEW', 'HOT', 'TOP'])
            badge_x = x + text_width - 30
            badge_y = y - 20
            
            if badge_y > 0 and badge_x > 0:
                draw.rectangle([badge_x - 5, badge_y - 2, badge_x + 35, badge_y + 18], 
                              fill=(255, 0, 0))
                draw.text((badge_x, badge_y), badge_text, fill=(255, 255, 255), font=badge_font)
            
            return self.batch_generator.pil_to_uint8_tensor(base_img)
            
        except Exception as e:
            logger.error(f"Error creating advertisement style: {e}")
            return None

    def generate_style_tensor(self, base_image_path, style):
        """Generate a single styled tensor"""
        if style == "youtube":
            return self.create_youtube_style(base_image_path)
        elif style == "social":
            return self.create_social_media_style(base_image_path)
        elif style == "trending":
            return self.create_trending_style(base_image_path)
        elif style == "news":
            return self.create_news_style(base_image_path)
        elif style == "advertisement":
            return self.create_advertisement_style(base_image_path)
        else:
            logger.error(f"Unknown style: {style}")
            return None

# ============================================================================
# LARGE-SCALE TENSOR BATCH GENERATOR
# ============================================================================

class LargeScaleTensorBatchManager:
    """
    Manages large-scale generation directly to tensor batches
    """
    
    def __init__(self, real_folder, synthetic_folder, output_folder, target_count=80000, images_per_batch=5000):
        self.real_folder = real_folder
        self.synthetic_folder = synthetic_folder
        self.output_folder = Path(output_folder)
        self.target_count = target_count
        self.images_per_batch = images_per_batch
        
        # Initialize generator
        self.generator = SemiSyntheticTensorGenerator()
        
        # Get available images
        self.real_images = self._get_images(real_folder) if real_folder else []
        self.synthetic_images = self._get_images(synthetic_folder) if synthetic_folder else []
        self.all_images = self.real_images + self.synthetic_images
        
        logger.info(f"Available images: {len(self.real_images)} real, {len(self.synthetic_images)} synthetic")
        
        if not self.all_images:
            logger.error("No source images available!")
            raise ValueError("No source images found")
        
        # Calculate number of batches needed
        self.num_batches = (target_count + images_per_batch - 1) // images_per_batch
        
        logger.info(f"Will create {self.num_batches} .pt files with {images_per_batch} images each")
        
    def _get_images(self, folder):
        """Get all image files from folder"""
        if not folder or not os.path.exists(folder):
            return []
        
        images = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        for ext in extensions:
            images.extend(Path(folder).glob(ext))
        return [str(img) for img in images]
    
    def generate_tensor_batches(self):
        """
        Generate images and save them directly as tensor batches
        Each .pt file contains exactly 5000 uint8 image tensors
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting generation of {self.target_count} images in {self.num_batches} .pt files")
        logger.info(f"Each .pt file will contain {self.images_per_batch} uint8 image tensors")
        
        total_generated = 0
        total_saved_batches = 0
        start_time = time.time()
        
        # Generate progress log
        progress_log = {
            'start_time': datetime.now().isoformat(),
            'target_count': self.target_count,
            'images_per_batch': self.images_per_batch,
            'num_batches': self.num_batches,
            'tensor_format': 'uint8',
            'batches': []
        }
        
        for batch_idx in range(self.num_batches):
            batch_start = batch_idx * self.images_per_batch
            batch_end = min(batch_start + self.images_per_batch, self.target_count)
            current_batch_size = batch_end - batch_start
            
            logger.info(f"Generating batch {batch_idx + 1}/{self.num_batches} ({current_batch_size} images)")
            batch_start_time = time.time()
            
            # Generate tensors for this batch
            batch_tensors = []
            batch_metadata = {
                'batch_index': batch_idx,
                'target_batch_size': current_batch_size,
                'styles_used': [],
                'source_images_used': []
            }
            
            with tqdm(total=current_batch_size, desc=f"Batch {batch_idx + 1}") as pbar:
                for i in range(current_batch_size):
                    # Choose style and source image
                    style = random.choice(self.generator.styles)
                    
                    # Prefer real images (60% chance if available)
                    if self.real_images and random.random() < 0.6:
                        base_img = random.choice(self.real_images)
                        source_type = "real"
                    else:
                        base_img = random.choice(self.all_images)
                        source_type = "synthetic" if base_img in self.synthetic_images else "real"
                    
                    # Generate styled tensor
                    tensor = self.generator.generate_style_tensor(base_img, style)
                    
                    if tensor is not None:
                        batch_tensors.append(tensor)
                        batch_metadata['styles_used'].append(style)
                        batch_metadata['source_images_used'].append({
                            'path': str(Path(base_img).name),
                            'type': source_type,
                            'style': style
                        })
                    else:
                        logger.warning(f"Failed to generate tensor for {base_img} with style {style}")
                    
                    pbar.update(1)
            
            # Save batch as .pt file
            if batch_tensors:
                batch_metadata['actual_batch_size'] = len(batch_tensors)
                
                output_filename = f"semi_synthetic_batch_{batch_idx+1:04d}_of_{self.num_batches:04d}.pt"
                output_path = self.output_folder / output_filename
                
                success = self.generator.batch_generator.save_tensor_batch(
                    batch_tensors, 
                    output_path, 
                    batch_metadata
                )
                
                if success:
                    total_saved_batches += 1
                    total_generated += len(batch_tensors)
                    
                    batch_time = time.time() - batch_start_time
                    
                    # Log batch results
                    batch_info = {
                        'batch_idx': batch_idx + 1,
                        'filename': output_filename,
                        'target_size': current_batch_size,
                        'actual_size': len(batch_tensors),
                        'time_seconds': batch_time,
                        'images_per_second': len(batch_tensors) / batch_time if batch_time > 0 else 0
                    }
                    progress_log['batches'].append(batch_info)
                    
                    logger.info(f"Batch {batch_idx + 1} complete: {len(batch_tensors)} tensors saved")
                    logger.info(f"Batch time: {batch_time:.1f}s ({len(batch_tensors)/batch_time:.1f} img/s)")
                else:
                    logger.error(f"Failed to save batch {batch_idx + 1}")
            else:
                logger.warning(f"No valid tensors generated for batch {batch_idx + 1}")
            
            # Clear memory
            del batch_tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save progress log
            progress_log['end_time'] = datetime.now().isoformat()
            progress_log['total_generated'] = total_generated
            progress_log['total_batches_saved'] = total_saved_batches
            progress_log['total_time_seconds'] = time.time() - start_time
            
            try:
                with open(self.output_folder / 'generation_log.json', 'w') as f:
                    json.dump(progress_log, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save progress log: {e}")
        
        total_time = time.time() - start_time
        
        logger.info("TENSOR BATCH GENERATION COMPLETE!")
        logger.info(f"Total images generated: {total_generated}/{self.target_count}")
        logger.info(f"Total .pt files created: {total_saved_batches}/{self.num_batches}")
        logger.info(f"Total time: {total_time:.1f}s ({total_generated/total_time:.1f} img/s)")
        logger.info(f"Output folder: {self.output_folder}")
        
        # Final summary
        success_rate = (total_generated / self.target_count) * 100 if self.target_count > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        # List created files
        pt_files = list(self.output_folder.glob("*.pt"))
        total_size_mb = sum(f.stat().st_size for f in pt_files) / (1024 * 1024)
        
        logger.info(f"Created .pt files ({total_size_mb:.1f} MB total):")
        for pt_file in sorted(pt_files):
            file_size_mb = pt_file.stat().st_size / (1024 * 1024)
            logger.info(f"  - {pt_file.name} ({file_size_mb:.1f} MB)")
        
        return total_generated, total_saved_batches

# ============================================================================
# TENSOR BATCH UTILITIES
# ============================================================================

def load_tensor_batch(pt_file_path, device='cpu'):
    """
    Load a tensor batch file and return the tensor directly
    
    Args:
        pt_file_path: Path to the .pt file
        device: Device to load tensors to ('cpu', 'cuda', etc.)
        
    Returns:
        torch.Tensor: Loaded tensor with shape (N, C, H, W)
    """
    try:
        # Load tensor directly (no dict wrapper)
        tensor = torch.load(pt_file_path, map_location=device)
        
        if torch.is_tensor(tensor):
            logger.info(f"Loaded tensor batch: {pt_file_path}")
            logger.info(f"  - Shape: {tensor.shape}")
            logger.info(f"  - Device: {tensor.device}")
            logger.info(f"  - Dtype: {tensor.dtype}")
            logger.info(f"  - Batch size: {tensor.shape[0]}")
            
            # Verify uint8 format
            if tensor.dtype != torch.uint8:
                logger.warning(f"Expected uint8, got {tensor.dtype}")
            
            return tensor
        else:
            logger.warning(f"Expected tensor, got {type(tensor)} in {pt_file_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading {pt_file_path}: {e}")
        return None

def load_tensor_batch_with_metadata(pt_file_path, device='cpu'):
    """
    Load tensor batch and its associated metadata JSON file
    
    Args:
        pt_file_path: Path to the .pt file
        device: Device to load tensors to ('cpu', 'cuda', etc.)
        
    Returns:
        tuple: (tensor, metadata_dict) or (None, None) if failed
    """
    try:
        # Load tensor
        tensor = load_tensor_batch(pt_file_path, device)
        if tensor is None:
            return None, None
        
        # Load metadata
        metadata_path = Path(pt_file_path).with_suffix('.json')
        metadata = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path.name}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        else:
            logger.info(f"No metadata file found for {pt_file_path}")
        
        return tensor, metadata
        
    except Exception as e:
        logger.error(f"Error loading tensor with metadata: {e}")
        return None, None

def extract_images_from_tensor_batch(pt_file_path, output_folder, max_images=None, save_format='jpg'):
    """
    Extract individual images from a uint8 tensor batch file
    
    Args:
        pt_file_path: Path to the .pt file
        output_folder: Folder to save extracted images
        max_images: Maximum number of images to extract (all if None)
        save_format: Image format to save ('jpg', 'png', etc.)
        
    Returns:
        int: Number of images extracted
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load the tensor
    images_tensor = load_tensor_batch(pt_file_path)
    if images_tensor is None:
        return 0
    
    if not torch.is_tensor(images_tensor):
        logger.error("No tensor data found")
        return 0
    
    num_images = images_tensor.shape[0]
    if max_images:
        num_images = min(num_images, max_images)
    
    logger.info(f"Extracting {num_images} images from uint8 tensor batch")
    
    extracted_count = 0
    batch_name = Path(pt_file_path).stem
    
    for i in tqdm(range(num_images), desc="Extracting images"):
        try:
            # Get single image tensor (C, H, W) in uint8 format
            img_tensor = images_tensor[i]
            
            # Verify uint8 format
            if img_tensor.dtype != torch.uint8:
                logger.warning(f"Expected uint8, got {img_tensor.dtype}, converting...")
                img_tensor = img_tensor.clamp(0, 255).byte()
            
            # Convert from (C, H, W) to (H, W, C) for PIL
            img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
            
            # Create PIL Image (uint8 array is already in [0,255] range)
            pil_image = Image.fromarray(img_array)
            
            # Save image
            filename = f"{batch_name}_image_{i+1:04d}.{save_format}"
            output_path = Path(output_folder) / filename
            
            # Convert to RGB if necessary for JPEG
            if pil_image.mode in ['RGBA', 'LA', 'P'] and save_format.lower() == 'jpg':
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGB')
                else:
                    rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ['RGBA', 'LA'] else None)
                    pil_image = rgb_image
            
            if save_format.lower() == 'jpg':
                pil_image.save(output_path, 'JPEG', quality=85, optimize=True)
            else:
                pil_image.save(output_path, save_format.upper())
            
            extracted_count += 1
            
        except Exception as e:
            logger.error(f"Error extracting image {i}: {e}")
            continue
    
    logger.info(f"Extracted {extracted_count} images to {output_folder}")
    return extracted_count

def verify_tensor_batch_format(pt_file_path):
    """
    Verify that a tensor batch file has the correct format (uint8, shape, etc.)
    
    Args:
        pt_file_path: Path to the .pt file
        
    Returns:
        dict: Verification results
    """
    try:
        tensor = torch.load(pt_file_path, map_location='cpu')
        
        verification = {
            'file_path': str(pt_file_path),
            'file_size_mb': Path(pt_file_path).stat().st_size / (1024 * 1024),
            'is_tensor': False,
            'correct_dtype': False,
            'correct_shape': False,
            'batch_size': 0,
            'issues': []
        }
        
        if torch.is_tensor(tensor):
            verification['is_tensor'] = True
            verification['batch_size'] = tensor.shape[0]
            
            # Check dtype
            if tensor.dtype == torch.uint8:
                verification['correct_dtype'] = True
            else:
                verification['issues'].append(f"Expected uint8, got {tensor.dtype}")
            
            # Check shape (should be N, C, H, W)
            if len(tensor.shape) == 4:
                verification['correct_shape'] = True
                verification['tensor_shape'] = list(tensor.shape)
                
                # Check if reasonable image dimensions
                if tensor.shape[1] not in [1, 3, 4]:  # Channels
                    verification['issues'].append(f"Unusual channel count: {tensor.shape[1]}")
                if tensor.shape[2] < 32 or tensor.shape[3] < 32:
                    verification['issues'].append(f"Very small image size: {tensor.shape[2]}x{tensor.shape[3]}")
            else:
                verification['issues'].append(f"Expected 4D tensor (N,C,H,W), got shape {tensor.shape}")
            
            # Check for associated metadata file
            metadata_path = Path(pt_file_path).with_suffix('.json')
            if metadata_path.exists():
                verification['has_metadata'] = True
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    verification['metadata'] = metadata
                except:
                    verification['issues'].append("Metadata file exists but could not be loaded")
            else:
                verification['has_metadata'] = False
            
        else:
            verification['issues'].append(f"Expected tensor, got {type(tensor)}")
        
        verification['is_valid'] = (verification['is_tensor'] and 
                                  verification['correct_dtype'] and 
                                  verification['correct_shape'] and 
                                  len(verification['issues']) == 0)
        
        return verification
        
    except Exception as e:
        return {
            'file_path': str(pt_file_path),
            'is_tensor': False,
            'correct_dtype': False,
            'correct_shape': False,
            'batch_size': 0,
            'is_valid': False,
            'issues': [f"Error loading file: {e}"]
        }

def convert_existing_images_to_tensor_batches(image_folder, output_folder, images_per_batch=5000, tensor_size=(224, 224)):
    """
    Convert existing images to uint8 tensor batches (.pt files)
    
    Args:
        image_folder: Folder containing source images
        output_folder: Folder to save .pt files
        images_per_batch: Number of images per .pt file (default: 5000)
        tensor_size: Target size for resizing images (width, height)
        
    Returns:
        int: Number of .pt files created
    """
    if not os.path.exists(image_folder):
        logger.error(f"Image folder {image_folder} does not exist")
        return 0
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(Path(image_folder).glob(ext))
        all_images.extend(Path(image_folder).glob(ext.upper()))
    
    if not all_images:
        logger.error(f"No images found in {image_folder}")
        return 0
    
    all_images = sorted(list(set(all_images)))  # Remove duplicates and sort
    total_images = len(all_images)
    num_batches = (total_images + images_per_batch - 1) // images_per_batch
    
    logger.info(f"Converting {total_images} images to {num_batches} uint8 tensor batches")
    logger.info(f"Images per batch: {images_per_batch}")
    logger.info(f"Target tensor size: {tensor_size}")
    
    # Initialize batch generator
    batch_generator = TensorBatchGenerator(tensor_size, images_per_batch)
    
    created_batches = 0
    total_processed = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * images_per_batch
        end_idx = min(start_idx + images_per_batch, total_images)
        batch_files = all_images[start_idx:end_idx]
        current_batch_size = len(batch_files)
        
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} - {current_batch_size} images")
        
        # Process images in current batch
        batch_tensors = []
        successful_files = []
        
        for img_path in tqdm(batch_files, desc=f"Batch {batch_idx + 1}"):
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB to ensure consistency
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Convert to uint8 tensor
                    tensor = batch_generator.pil_to_uint8_tensor(img)
                    if tensor is not None:
                        batch_tensors.append(tensor)
                        successful_files.append(str(img_path.name))
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        if not batch_tensors:
            logger.warning(f"No valid images in batch {batch_idx + 1}")
            continue
        
        # Create metadata
        metadata = {
            'batch_index': batch_idx,
            'target_batch_size': current_batch_size,
            'actual_batch_size': len(batch_tensors),
            'total_batches': num_batches,
            'image_size': tensor_size,
            'pixel_format': 'uint8',
            'value_range': '[0, 255]',
            'creation_time': datetime.now().isoformat(),
            'source_folder': str(image_folder),
            'image_files': successful_files
        }
        
        # Save .pt file
        output_filename = f"converted_batch_{batch_idx+1:04d}_of_{num_batches:04d}.pt"
        output_path = Path(output_folder) / output_filename
        
        success = batch_generator.save_tensor_batch(batch_tensors, output_path, metadata)
        
        if success:
            created_batches += 1
            total_processed += len(batch_tensors)
            logger.info(f"Batch {batch_idx + 1} complete: {len(batch_tensors)} images processed")
        else:
            logger.error(f"Failed to save batch {batch_idx + 1}")
        
        # Clear memory
        del batch_tensors
        gc.collect()
    
    logger.info(f"Conversion complete!")
    logger.info(f"Created {created_batches} .pt files with {total_processed} total images")
    
    return created_batches

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def create_sample_images(output_folder="sample_images", num_images=20):
    """Create sample images for testing if no source images are available"""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_images} sample images in {output_folder}")
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    patterns = ['solid', 'gradient', 'stripes', 'checkerboard']
    
    for i in range(num_images):
        # Create base image
        img = Image.new('RGB', (400, 300), random.choice(colors))
        draw = ImageDraw.Draw(img)
        
        pattern = random.choice(patterns)
        
        if pattern == 'gradient':
            # Simple gradient effect
            for y in range(300):
                color_val = int(255 * (y / 300))
                draw.line([(0, y), (400, y)], fill=(color_val, color_val, 255 - color_val))
        
        elif pattern == 'stripes':
            # Vertical stripes
            stripe_width = 40
            for x in range(0, 400, stripe_width * 2):
                draw.rectangle([x, 0, x + stripe_width, 300], fill=(255, 255, 255))
        
        elif pattern == 'checkerboard':
            # Checkerboard pattern
            size = 50
            for x in range(0, 400, size):
                for y in range(0, 300, size):
                    if (x // size + y // size) % 2 == 0:
                        draw.rectangle([x, y, x + size, y + size], fill=(255, 255, 255))
        
        # Add text
        try:
            font = ImageFont.load_default()
            text = f"Sample Image {i+1}"
            draw.text((50, 130), text, fill=(0, 0, 0), font=font)
        except:
            draw.text((50, 130), f"Sample {i+1}", fill=(0, 0, 0))
        
        # Add some shapes
        draw.ellipse([300, 50, 350, 100], fill=(255, 255, 0))
        draw.rectangle([320, 200, 370, 250], fill=(255, 0, 255))
        
        # Save image
        img.save(Path(output_folder) / f"sample_{i+1:03d}.jpg", 'JPEG', quality=85)
    
    logger.info(f"Created {num_images} sample images")
    return output_folder

def setup_and_generate_50k_tensor_batches():
    """
    Complete setup and generation pipeline for 50K images saved as tensor batches
    """
    
    logger.info("LARGE-SCALE SEMI-SYNTHETIC TENSOR BATCH GENERATION (50K Images)")
    logger.info("=" * 80)
    logger.info("Output format: .pt files with 5000 uint8 image tensors each")
    
    # Step 1: Find or convert source images
    logger.info("Step 1: Preparing source images...")
    
    real_folder = None
    synthetic_folder = None
    
    # Check for existing image folders first
    image_paths = [
        ("images/real", "real"),
        ("images/synthetic", "synthetic"),
        ("real_images", "real"),
        ("synthetic_images", "synthetic"),
        ("datasets/real", "real"),
        ("datasets/synthetic", "synthetic"),
        ("sample_images", "real")  # Added sample images as fallback
    ]
    
    for image_path, image_type in image_paths:
        if os.path.exists(image_path):
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
                images.extend(Path(image_path).glob(ext))
            
            if images:
                logger.info(f"Found {len(images)} existing {image_type} images in {image_path}")
                if image_type == "real" and not real_folder:
                    real_folder = image_path
                elif image_type == "synthetic" and not synthetic_folder:
                    synthetic_folder = image_path
    
    # If still no images found, create sample images
    if not real_folder and not synthetic_folder:
        logger.info("No existing images found. Creating sample images for demonstration...")
        real_folder = create_sample_images("sample_images", 50)
    
    # Step 2: Generate tensor batches directly
    logger.info("Step 2: Generating 80,000 images as uint8 tensor batches...")
    
    try:
        manager = LargeScaleTensorBatchManager(
            real_folder=real_folder,
            synthetic_folder=synthetic_folder,
            output_folder="semi_synthetic_tensor_batches_80k",
            target_count=80000,
            images_per_batch=5000
        )
        
        # Start generation
        total_generated, total_batches = manager.generate_tensor_batches()
        
        if total_batches > 0:
            logger.info(f"SUCCESS: Generated {total_generated} images in {total_batches} .pt files")
            logger.info(f"Each .pt file contains up to 5000 uint8 image tensors")
            logger.info(f"Output folder: semi_synthetic_tensor_batches_80k/")
            return total_generated
        else:
            logger.error("No tensor batches were created")
            return 0
            
    except ValueError as e:
        logger.error(f"Failed to initialize manager: {e}")
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 0

def test_tensor_batch_functionality():
    """
    Test the tensor batch functionality with a small sample
    """
    logger.info("TESTING TENSOR BATCH FUNCTIONALITY")
    logger.info("=" * 50)
    
    # Create test images folder
    test_folder = create_sample_images("test_images", 15)
    
    # Test tensor batch generation
    logger.info("Testing tensor batch generation...")
    
    try:
        manager = LargeScaleTensorBatchManager(
            real_folder=str(test_folder),
            synthetic_folder=None,
            output_folder="test_tensor_batches",
            target_count=12,
            images_per_batch=5
        )
        
        total_generated, total_batches = manager.generate_tensor_batches()
        
        logger.info(f"Test generation complete: {total_generated} images in {total_batches} batches")
        
        # Test loading and verification
        if total_batches > 0:
            batch_files = list(Path("test_tensor_batches").glob("*.pt"))
            
            for batch_file in batch_files:
                logger.info(f"Verifying {batch_file.name}...")
                verification = verify_tensor_batch_format(batch_file)
                
                if verification['is_valid']:
                    logger.info(f"  ✓ Valid batch: {verification['batch_size']} images, {verification['file_size_mb']:.1f}MB")
                else:
                    logger.warning(f"  ✗ Invalid batch: {verification['issues']}")
            
            # Test extraction
            if batch_files:
                logger.info(f"Testing image extraction from {batch_files[0].name}...")
                extracted = extract_images_from_tensor_batch(
                    batch_files[0], 
                    "test_extracted_images", 
                    max_images=3
                )
                logger.info(f"Extracted {extracted} test images")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Semi-Synthetic Image Generator with Direct Tensor Batching')
    parser.add_argument('--test', action='store_true', help='Run functionality test')
    parser.add_argument('--generate', action='store_true', help='Generate 80K images as tensor batches')
    parser.add_argument('--convert', help='Convert existing images folder to tensor batches')
    parser.add_argument('--output', '-o', default='tensor_batches', help='Output folder for tensor batches')
    parser.add_argument('--extract', help='Extract images from a .pt file for verification')
    parser.add_argument('--extract-output', default='extracted_images', help='Output folder for extracted images')
    parser.add_argument('--verify', help='Verify format of a .pt file')
    parser.add_argument('--batch-size', type=int, default=5000, help='Images per .pt file')
    parser.add_argument('--target-count', type=int, default=80000, help='Target number of images to generate')
    parser.add_argument('--create-samples', action='store_true', help='Create sample images for testing')
    
    args = parser.parse_args()
    
    if args.create_samples:
        folder = create_sample_images("sample_images", 50)
        logger.info(f"Created sample images in {folder}")
        exit(0)
    
    if args.test:
        success = test_tensor_batch_functionality()
        if success:
            logger.info("All tests passed!")
        else:
            logger.error("Tests failed!")
        exit(0 if success else 1)
    
    if args.convert:
        result = convert_existing_images_to_tensor_batches(
            args.convert, 
            args.output, 
            args.batch_size
        )
        if result > 0:
            logger.info(f"Successfully converted images to {result} tensor batches")
        else:
            logger.error("Conversion failed")
        exit(0 if result > 0 else 1)
    
    if args.extract:
        count = extract_images_from_tensor_batch(args.extract, args.extract_output)
        if count > 0:
            logger.info(f"Successfully extracted {count} images")
        else:
            logger.error("Extraction failed")
        exit(0 if count > 0 else 1)
    
    if args.verify:
        verification = verify_tensor_batch_format(args.verify)
        logger.info(f"Verification results for {args.verify}:")
        logger.info(f"  Valid: {verification['is_valid']}")
        if verification['is_valid']:
            logger.info(f"  Batch size: {verification['batch_size']}")
            logger.info(f"  Tensor shape: {verification.get('tensor_shape', 'N/A')}")
            logger.info(f"  File size: {verification['file_size_mb']:.1f} MB")
        else:
            logger.info(f"  Issues: {verification['issues']}")
        exit(0)
    
    if args.generate:
        result = setup_and_generate_50k_tensor_batches()
        if result > 0:
            logger.info("Generation completed successfully!")
        else:
            logger.error("Generation failed!")
        exit(0 if result > 0 else 1)
    
    # Default: show help
    parser.print_help()