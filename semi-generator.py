import torch
import numpy as np
import cv2
import os
import glob
import random
import math
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageOps
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional, Dict, Any
import json
from pathlib import Path
import requests
from io import BytesIO
import base64

class EnhancedSemiSyntheticProcessor:
    """
    Enhanced processor for creating semi-synthetic images with diverse AI-generated modifications
    Output: torch.uint8 tensors with shape (3, 224, 224) and range [0, 255]
    """
    
    def __init__(self):
        self.target_size = (224, 224)
        self.supported_formats = ['.pt', '.pth', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.min_difference_threshold = 25.0
        
        # Expanded shape types
        self.shape_types = [
            'circles', 'rectangles', 'triangles', 'stars', 'polygons', 
            'ellipses', 'diamonds', 'hexagons', 'crescents', 'arrows',
            'hearts', 'crosses', 'spirals', 'bezier_curves', 'wave_lines'
        ]
        
        # Expanded effect types
        self.effect_types = [
            'lens_flare', 'light_rays', 'spotlight', 'aurora', 'lightning',
            'rainbow', 'sun_rays', 'laser_beams', 'glow_effects', 'halo'
        ]
        
        # Expanded particle systems
        self.particle_types = [
            'sparkles', 'bubbles', 'snow', 'rain', 'fireflies', 'dust',
            'confetti', 'leaves', 'petals', 'embers', 'stars', 'crystals'
        ]
        
        # New blur effects
        self.blur_types = [
            'gaussian', 'motion', 'radial', 'zoom', 'selective', 'bokeh'
        ]
        
        # Text styles
        self.text_styles = [
            'watermark', 'graffiti', 'neon', 'embossed', 'shadow', 'outline'
        ]
        
        # Pattern types
        self.pattern_types = [
            'stripes', 'checkerboard', 'dots', 'waves', 'spirals', 'fractals'
        ]
        
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array with proper format"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        if tensor.dim() == 3:
            tensor = tensor.detach().cpu()
            
            if tensor.shape[0] == 3:
                img = tensor.permute(1, 2, 0).numpy()
            elif tensor.shape[2] == 3:
                img = tensor.numpy()
            else:
                raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
            
            if tensor.dtype == torch.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            elif img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255.0).astype(np.uint8)
            elif img.max() <= 1.0 and img.min() >= -1.0:
                img = ((img + 1.0) * 127.5).astype(np.uint8)
            elif img.min() >= 0 and img.max() <= 255:
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            
            return img
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")
    
    def numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy array back to PyTorch tensor in CHW format as uint8"""
        if len(img.shape) == 3 and img.shape[2] == 3:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            if img.min() < 0 or img.max() > 255:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            tensor = torch.from_numpy(img.copy()).permute(2, 0, 1)
            
            if tensor.dtype != torch.uint8:
                tensor = tensor.to(torch.uint8)
            
            assert tensor.dtype == torch.uint8
            assert tensor.shape == (3, 224, 224)
            assert tensor.min().item() >= 0 and tensor.max().item() <= 255
            
            return tensor
        else:
            raise ValueError(f"Expected 3D numpy array with shape [H,W,3], got shape {img.shape}")

    def verify_transformation(self, original: np.ndarray, transformed: np.ndarray) -> Tuple[bool, float]:
        """Verify that transformation actually occurred but preserved content"""
        if original.shape != transformed.shape:
            return False, 0.0
            
        if np.std(transformed) < 5:
            return False, 0.0
        
        diff = np.mean((original.astype(float) - transformed.astype(float)) ** 2)
        is_transformed = self.min_difference_threshold < diff < 10000
        
        return is_transformed, diff

    def load_mixed_files(self, directory: str, count: int) -> Tuple[List[torch.Tensor], List[str]]:
        """Load both tensor and image files from directory with better error handling"""
        all_files = []
        for ext in self.supported_formats:
            all_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        
        if not all_files:
            raise ValueError(f"No supported files found in {directory}")
        
        all_files.sort()
        loaded_tensors = []
        loaded_files = []
        total_loaded = 0
        
        for file_path in all_files:
            if total_loaded >= count:
                break
                
            try:
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in ['.pt', '.pth']:
                    tensor_batch = torch.load(file_path, map_location='cpu')
                    
                    if len(tensor_batch.shape) == 4:
                        if tensor_batch.shape[1] == 3 and tensor_batch.shape[2] == 224 and tensor_batch.shape[3] == 224:
                            remaining_needed = count - total_loaded
                            take_count = min(tensor_batch.shape[0], remaining_needed)
                            batch_subset = tensor_batch[:take_count]
                            loaded_tensors.append(batch_subset)
                            loaded_files.append(file_path)
                            total_loaded += take_count
                        elif tensor_batch.shape[3] == 3 and tensor_batch.shape[1] == 224 and tensor_batch.shape[2] == 224:
                            tensor_batch = tensor_batch.permute(0, 3, 1, 2)
                            remaining_needed = count - total_loaded
                            take_count = min(tensor_batch.shape[0], remaining_needed)
                            batch_subset = tensor_batch[:take_count]
                            loaded_tensors.append(batch_subset)
                            loaded_files.append(file_path)
                            total_loaded += take_count
                    elif len(tensor_batch.shape) == 3:
                        if tensor_batch.shape[0] == 3:
                            tensor_batch = tensor_batch.unsqueeze(0)
                        elif tensor_batch.shape[2] == 3:
                            tensor_batch = tensor_batch.permute(2, 0, 1).unsqueeze(0)
                        
                        loaded_tensors.append(tensor_batch)
                        loaded_files.append(file_path)
                        total_loaded += 1
                
                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    try:
                        img = Image.open(file_path).convert('RGB')
                        img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img)
                        
                        if img_array.std() < 1:
                            continue
                        
                        tensor = self.numpy_to_tensor(img_array).unsqueeze(0)
                        loaded_tensors.append(tensor)
                        loaded_files.append(file_path)
                        total_loaded += 1
                        
                    except Exception:
                        continue
                    
            except Exception:
                continue
        
        if total_loaded == 0:
            raise ValueError("No valid files could be loaded")
        
        return loaded_tensors, loaded_files

    def add_diverse_geometric_shapes(self, img: np.ndarray, num_shapes: int = 3) -> np.ndarray:
        """Add diverse geometric shapes with various styles and effects"""
        result = img.copy()
        overlay = result.copy()
        height, width = result.shape[:2]
        
        for _ in range(num_shapes):
            x = random.randint(width//6, 5*width//6)
            y = random.randint(height//6, 5*height//6)
            size = random.randint(min(width, height)//15, min(width, height)//5)
            
            # More diverse colors including gradients
            if random.random() < 0.3:  # Gradient shapes
                color1 = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                color2 = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            else:  # Solid colors
                color = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
                color1 = color2 = color
            
            shape_type = random.choice(self.shape_types)
            
            if shape_type == 'circles':
                cv2.circle(overlay, (x, y), size, color1, -1)
            elif shape_type == 'rectangles':
                cv2.rectangle(overlay, (x-size, y-size), (x+size, y+size), color1, -1)
            elif shape_type == 'triangles':
                points = np.array([[x, y-size], [x-size, y+size], [x+size, y+size]], np.int32)
                cv2.fillPoly(overlay, [points], color1)
            elif shape_type == 'stars':
                self._draw_star(overlay, x, y, size, color1)
            elif shape_type == 'polygons':
                self._draw_polygon(overlay, x, y, size, color1, random.randint(5, 8))
            elif shape_type == 'ellipses':
                cv2.ellipse(overlay, (x, y), (size, size//2), random.randint(0, 180), 0, 360, color1, -1)
            elif shape_type == 'diamonds':
                self._draw_diamond(overlay, x, y, size, color1)
            elif shape_type == 'hexagons':
                self._draw_polygon(overlay, x, y, size, color1, 6)
            elif shape_type == 'crescents':
                self._draw_crescent(overlay, x, y, size, color1)
            elif shape_type == 'arrows':
                self._draw_arrow(overlay, x, y, size, color1)
            elif shape_type == 'hearts':
                self._draw_heart(overlay, x, y, size, color1)
            elif shape_type == 'crosses':
                self._draw_cross(overlay, x, y, size, color1)
            elif shape_type == 'spirals':
                self._draw_spiral(overlay, x, y, size, color1)
            elif shape_type == 'bezier_curves':
                self._draw_bezier_curve(overlay, x, y, size, color1)
            elif shape_type == 'wave_lines':
                self._draw_wave_line(overlay, x, y, size, color1)
        
        # Apply various blending modes
        alpha = random.uniform(0.2, 0.5)
        blend_mode = random.choice(['normal', 'multiply', 'screen', 'overlay'])
        
        if blend_mode == 'normal':
            result = cv2.addWeighted(result, 1-alpha, overlay, alpha, 0)
        elif blend_mode == 'multiply':
            result = (result.astype(float) * overlay.astype(float) / 255.0).astype(np.uint8)
        elif blend_mode == 'screen':
            result = (255 - ((255 - result.astype(float)) * (255 - overlay.astype(float)) / 255.0)).astype(np.uint8)
        elif blend_mode == 'overlay':
            mask = overlay > 128
            result[mask] = (2 * result[mask].astype(float) * overlay[mask].astype(float) / 255.0).astype(np.uint8)[mask]
        
        return result

    def _draw_star(self, img, x, y, size, color):
        """Draw a star shape"""
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            if i % 2 == 0:
                r = size
            else:
                r = size // 2
            px = int(x + r * math.cos(angle - math.pi/2))
            py = int(y + r * math.sin(angle - math.pi/2))
            points.append([px, py])
        cv2.fillPoly(img, [np.array(points, np.int32)], color)

    def _draw_polygon(self, img, x, y, size, color, sides):
        """Draw a regular polygon"""
        points = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            px = int(x + size * math.cos(angle))
            py = int(y + size * math.sin(angle))
            points.append([px, py])
        cv2.fillPoly(img, [np.array(points, np.int32)], color)

    def _draw_diamond(self, img, x, y, size, color):
        """Draw a diamond shape"""
        points = np.array([[x, y-size], [x+size, y], [x, y+size], [x-size, y]], np.int32)
        cv2.fillPoly(img, [points], color)

    def _draw_crescent(self, img, x, y, size, color):
        """Draw a crescent shape"""
        cv2.circle(img, (x, y), size, color, -1)
        cv2.circle(img, (x + size//3, y), size, (0, 0, 0), -1)

    def _draw_arrow(self, img, x, y, size, color):
        """Draw an arrow shape"""
        points = np.array([
            [x-size, y], [x, y-size//2], [x, y-size//4],
            [x+size, y-size//4], [x+size, y+size//4],
            [x, y+size//4], [x, y+size//2]
        ], np.int32)
        cv2.fillPoly(img, [points], color)

    def _draw_heart(self, img, x, y, size, color):
        """Draw a heart shape"""
        # Simple heart approximation using circles and triangle
        cv2.circle(img, (x-size//3, y-size//3), size//2, color, -1)
        cv2.circle(img, (x+size//3, y-size//3), size//2, color, -1)
        points = np.array([[x-size//2, y], [x, y+size], [x+size//2, y]], np.int32)
        cv2.fillPoly(img, [points], color)

    def _draw_cross(self, img, x, y, size, color):
        """Draw a cross shape"""
        thickness = size // 4
        cv2.rectangle(img, (x-thickness, y-size), (x+thickness, y+size), color, -1)
        cv2.rectangle(img, (x-size, y-thickness), (x+size, y+thickness), color, -1)

    def _draw_spiral(self, img, x, y, size, color):
        """Draw a spiral shape"""
        points = []
        for i in range(0, 360*3, 5):
            angle = math.radians(i)
            r = (i / (360*3)) * size
            px = int(x + r * math.cos(angle))
            py = int(y + r * math.sin(angle))
            points.append((px, py))
        
        for i in range(len(points)-1):
            cv2.line(img, points[i], points[i+1], color, 3)

    def _draw_bezier_curve(self, img, x, y, size, color):
        """Draw a bezier curve"""
        p0 = (x - size, y)
        p1 = (x - size//2, y - size)
        p2 = (x + size//2, y + size)
        p3 = (x + size, y)
        
        points = []
        for t in np.linspace(0, 1, 50):
            px = int((1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0])
            py = int((1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1])
            points.append((px, py))
        
        for i in range(len(points)-1):
            cv2.line(img, points[i], points[i+1], color, 4)

    def _draw_wave_line(self, img, x, y, size, color):
        """Draw a wave line"""
        points = []
        for i in range(-size, size, 3):
            px = x + i
            py = int(y + size//3 * math.sin(i * math.pi / (size//4)))
            points.append((px, py))
        
        for i in range(len(points)-1):
            cv2.line(img, points[i], points[i+1], color, 3)

    def add_advanced_light_effects(self, img: np.ndarray) -> np.ndarray:
        """Add advanced lighting effects with more variety"""
        result = img.copy().astype(np.float32)
        height, width = result.shape[:2]
        
        effect_type = random.choice(self.effect_types)
        
        if effect_type == 'lens_flare':
            result = self._add_lens_flare(result, width, height)
        elif effect_type == 'light_rays':
            result = self._add_light_rays(result, width, height)
        elif effect_type == 'spotlight':
            result = self._add_spotlight(result, width, height)
        elif effect_type == 'aurora':
            result = self._add_aurora_effect(result, width, height)
        elif effect_type == 'lightning':
            result = self._add_lightning_effect(result, width, height)
        elif effect_type == 'rainbow':
            result = self._add_rainbow_effect(result, width, height)
        elif effect_type == 'sun_rays':
            result = self._add_sun_rays(result, width, height)
        elif effect_type == 'laser_beams':
            result = self._add_laser_beams(result, width, height)
        elif effect_type == 'glow_effects':
            result = self._add_glow_effects(result, width, height)
        elif effect_type == 'halo':
            result = self._add_halo_effect(result, width, height)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_lens_flare(self, img, width, height):
        """Add realistic lens flare"""
        flare_x = random.randint(width//4, 3*width//4)
        flare_y = random.randint(height//4, 3*height//4)
        
        # Multiple flare elements
        for i, r in enumerate(range(20, 100, 15)):
            intensity = 1.0 - (i * 0.2)
            color_boost = intensity * random.randint(30, 60)
            
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (flare_x, flare_y), r, 255, -1)
            
            # Random color tinting
            color_tint = [random.uniform(0.8, 1.2) for _ in range(3)]
            for c in range(3):
                boost = color_boost * color_tint[c]
                img[:, :, c] = np.where(mask > 0, 
                                      np.minimum(img[:, :, c] + boost, 255),
                                      img[:, :, c])
        return img

    def _add_light_rays(self, img, width, height):
        """Add dramatic light rays"""
        center_x = random.randint(width//4, 3*width//4)
        center_y = random.randint(height//4, 3*height//4)
        num_rays = random.randint(5, 12)
        
        overlay = img.copy()
        
        for i in range(num_rays):
            angle = (2 * math.pi * i / num_rays) + random.uniform(-0.3, 0.3)
            length = random.randint(width//2, width)
            end_x = int(center_x + length * math.cos(angle))
            end_y = int(center_y + length * math.sin(angle))
            
            # Create ray with thickness
            thickness = random.randint(2, 8)
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(150, 255))
            cv2.line(overlay.astype(np.uint8), (center_x, center_y), (end_x, end_y), color, thickness)
        
        alpha = random.uniform(0.15, 0.35)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_spotlight(self, img, width, height):
        """Add spotlight effect"""
        center_x = random.randint(width//3, 2*width//3)
        center_y = random.randint(height//3, 2*height//3)
        radius = random.randint(min(width, height)//4, min(width, height)//2)
        
        # Create spotlight mask
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Gradual falloff
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        spotlight_intensity = np.maximum(0, 1 - distances / (radius * 1.5))
        
        for c in range(3):
            img[:, :, c] = img[:, :, c] + spotlight_intensity * 40
        
        return img

    def _add_aurora_effect(self, img, width, height):
        """Add aurora-like effect"""
        overlay = np.zeros_like(img)
        
        # Create wavy aurora bands
        for band in range(3):
            y_offset = height // 4 + band * 30
            for x in range(width):
                wave_y = int(y_offset + 20 * math.sin(x * 0.02 + band))
                if 0 <= wave_y < height:
                    # Aurora colors (green, blue, purple)
                    if band == 0:
                        color = (0, random.randint(100, 200), random.randint(50, 150))
                    elif band == 1:
                        color = (random.randint(50, 150), 0, random.randint(100, 200))
                    else:
                        color = (random.randint(100, 200), random.randint(50, 150), 0)
                    
                    for dy in range(-10, 11):
                        if 0 <= wave_y + dy < height:
                            intensity = 1.0 - abs(dy) / 10.0
                            for c in range(3):
                                overlay[wave_y + dy, x, c] = color[c] * intensity
        
        alpha = random.uniform(0.3, 0.6)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_lightning_effect(self, img, width, height):
        """Add lightning bolt effect"""
        # Create jagged lightning path
        start_x = random.randint(0, width)
        start_y = 0
        end_x = random.randint(0, width)
        end_y = height
        
        points = [(start_x, start_y)]
        
        # Generate jagged path
        segments = 8
        for i in range(1, segments):
            progress = i / segments
            base_x = int(start_x + (end_x - start_x) * progress)
            base_y = int(start_y + (end_y - start_y) * progress)
            
            # Add randomness
            offset_x = random.randint(-30, 30)
            offset_y = random.randint(-10, 10)
            
            points.append((base_x + offset_x, base_y + offset_y))
        
        points.append((end_x, end_y))
        
        # Draw lightning with glow
        overlay = img.copy()
        for thickness in [8, 4, 2]:
            color_intensity = 255 - (thickness - 2) * 40
            color = (color_intensity, color_intensity, 255)
            
            for i in range(len(points)-1):
                cv2.line(overlay.astype(np.uint8), points[i], points[i+1], color, thickness)
        
        alpha = random.uniform(0.4, 0.7)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_rainbow_effect(self, img, width, height):
        """Add rainbow arc effect"""
        center_x = width // 2
        center_y = height + 50  # Arc from bottom
        
        colors = [
            (255, 0, 0), (255, 127, 0), (255, 255, 0),  # Red, Orange, Yellow
            (0, 255, 0), (0, 0, 255), (75, 0, 130), (148, 0, 211)  # Green, Blue, Indigo, Violet
        ]
        
        overlay = img.copy()
        
        for i, color in enumerate(colors):
            radius = 100 + i * 8
            cv2.ellipse(overlay.astype(np.uint8), (center_x, center_y), (radius, radius//2), 
                       0, 180, 360, color, 3)
        
        alpha = random.uniform(0.25, 0.45)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_sun_rays(self, img, width, height):
        """Add sun rays radiating from a point"""
        sun_x = random.randint(width//6, 5*width//6)
        sun_y = random.randint(height//6, 5*height//6)
        
        overlay = img.copy()
        num_rays = random.randint(12, 20)
        
        for i in range(num_rays):
            angle = 2 * math.pi * i / num_rays
            length = random.randint(width//3, width)
            
            # Create multiple rays with slight angle variations
            for j in range(3):
                angle_var = angle + random.uniform(-0.1, 0.1)
                end_x = int(sun_x + length * math.cos(angle_var))
                end_y = int(sun_y + length * math.sin(angle_var))
                
                thickness = max(1, 4 - j)
                brightness = 200 - j * 50
                color = (brightness, brightness, min(255, brightness + 50))
                
                cv2.line(overlay.astype(np.uint8), (sun_x, sun_y), (end_x, end_y), color, thickness)
        
        alpha = random.uniform(0.2, 0.4)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_laser_beams(self, img, width, height):
        """Add laser beam effects"""
        overlay = img.copy()
        num_beams = random.randint(2, 5)
        
        for _ in range(num_beams):
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            end_x = random.randint(0, width)
            end_y = random.randint(0, height)
            
            # Laser colors (bright and saturated)
            laser_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]
            color = random.choice(laser_colors)
            
            # Draw laser with glow effect
            for thickness in [6, 4, 2]:
                intensity = 1.0 - (thickness - 2) * 0.2
                glow_color = tuple(int(c * intensity) for c in color)
                cv2.line(overlay.astype(np.uint8), (start_x, start_y), (end_x, end_y), glow_color, thickness)
        
        alpha = random.uniform(0.3, 0.5)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_glow_effects(self, img, width, height):
        """Add various glow effects"""
        overlay = img.copy()
        
        # Add multiple glowing orbs
        num_orbs = random.randint(3, 7)
        for _ in range(num_orbs):
            x = random.randint(width//6, 5*width//6)
            y = random.randint(height//6, 5*height//6)
            
            # Glow colors
            glow_colors = [(255, 200, 100), (100, 255, 200), (200, 100, 255), 
                          (255, 100, 100), (100, 100, 255), (255, 255, 100)]
            color = random.choice(glow_colors)
            
            # Create glow effect with multiple circles
            for radius in range(5, 25, 3):
                intensity = 1.0 - (radius - 5) / 20.0
                glow_color = tuple(int(c * intensity) for c in color)
                cv2.circle(overlay.astype(np.uint8), (x, y), radius, glow_color, -1)
        
        alpha = random.uniform(0.25, 0.45)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_halo_effect(self, img, width, height):
        """Add halo/ring light effects"""
        center_x = random.randint(width//3, 2*width//3)
        center_y = random.randint(height//3, 2*height//3)
        
        overlay = img.copy()
        
        # Create multiple concentric halos
        for ring in range(3):
            radius = 30 + ring * 15
            thickness = 6 - ring * 2
            
            # Halo colors (bright and ethereal)
            halo_colors = [(255, 255, 200), (200, 255, 255), (255, 200, 255)]
            color = halo_colors[ring % len(halo_colors)]
            
            cv2.circle(overlay.astype(np.uint8), (center_x, center_y), radius, color, thickness)
        
        alpha = random.uniform(0.3, 0.5)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def add_diverse_particle_systems(self, img: np.ndarray, num_particles: int = 60) -> np.ndarray:
        """Add diverse particle effects"""
        result = img.copy()
        height, width = result.shape[:2]
        
        particle_type = random.choice(self.particle_types)
        
        if particle_type == 'sparkles':
            result = self._add_sparkle_particles(result, width, height, num_particles)
        elif particle_type == 'bubbles':
            result = self._add_bubble_particles(result, width, height, num_particles // 2)
        elif particle_type == 'snow':
            result = self._add_snow_particles(result, width, height, num_particles)
        elif particle_type == 'rain':
            result = self._add_rain_particles(result, width, height, num_particles)
        elif particle_type == 'fireflies':
            result = self._add_firefly_particles(result, width, height, num_particles // 3)
        elif particle_type == 'dust':
            result = self._add_dust_particles(result, width, height, num_particles * 2)
        elif particle_type == 'confetti':
            result = self._add_confetti_particles(result, width, height, num_particles)
        elif particle_type == 'leaves':
            result = self._add_leaf_particles(result, width, height, num_particles // 2)
        elif particle_type == 'petals':
            result = self._add_petal_particles(result, width, height, num_particles // 2)
        elif particle_type == 'embers':
            result = self._add_ember_particles(result, width, height, num_particles // 2)
        elif particle_type == 'stars':
            result = self._add_star_particles(result, width, height, num_particles // 3)
        elif particle_type == 'crystals':
            result = self._add_crystal_particles(result, width, height, num_particles // 4)
        
        return result

    def _add_sparkle_particles(self, img, width, height, num_particles):
        """Add sparkling particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(2, 5)
            
            # Bright sparkle colors
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            
            # Cross pattern for sparkles
            cv2.line(img, (max(0, x-size), y), (min(width-1, x+size), y), color, 2)
            cv2.line(img, (x, max(0, y-size)), (x, min(height-1, y+size)), color, 2)
            cv2.circle(img, (x, y), 1, color, -1)
        
        return img

    def _add_bubble_particles(self, img, width, height, num_particles):
        """Add bubble particles"""
        overlay = img.copy()
        
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(5, 15)
            
            # Semi-transparent bubble colors
            color = (random.randint(180, 255), random.randint(180, 255), random.randint(200, 255))
            
            # Draw bubble outline
            cv2.circle(overlay, (x, y), size, color, 2)
            
            # Add highlight
            highlight_x = x - size // 3
            highlight_y = y - size // 3
            cv2.circle(overlay, (highlight_x, highlight_y), size // 4, (255, 255, 255), -1)
        
        alpha = 0.4
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_snow_particles(self, img, width, height, num_particles):
        """Add snow particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(1, 4)
            
            # White/light blue snow colors
            color = (random.randint(240, 255), random.randint(240, 255), 255)
            
            cv2.circle(img, (x, y), size, color, -1)
            
            # Some snowflakes get additional detail
            if random.random() < 0.3 and size > 2:
                for angle in range(0, 360, 60):
                    dx = int(size * 1.5 * math.cos(math.radians(angle)))
                    dy = int(size * 1.5 * math.sin(math.radians(angle)))
                    if 0 <= x+dx < width and 0 <= y+dy < height:
                        cv2.line(img, (x, y), (x+dx, y+dy), color, 1)
        
        return img

    def _add_rain_particles(self, img, width, height, num_particles):
        """Add rain particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-20)
            length = random.randint(5, 15)
            
            # Rain colors (blue-gray)
            color = (random.randint(150, 200), random.randint(180, 220), random.randint(200, 255))
            
            # Diagonal rain lines
            angle_offset = random.randint(-5, 5)
            end_x = x + angle_offset
            end_y = y + length
            
            if 0 <= end_x < width and 0 <= end_y < height:
                cv2.line(img, (x, y), (end_x, end_y), color, 2)
        
        return img

    def _add_firefly_particles(self, img, width, height, num_particles):
        """Add firefly particles with glow"""
        overlay = img.copy()
        
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            
            # Warm firefly colors
            color = (random.randint(200, 255), random.randint(220, 255), random.randint(100, 150))
            
            # Glow effect
            for radius in range(8, 2, -1):
                intensity = 1.0 - (8 - radius) / 6.0
                glow_color = tuple(int(c * intensity) for c in color)
                cv2.circle(overlay, (x, y), radius, glow_color, -1)
        
        alpha = 0.5
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_dust_particles(self, img, width, height, num_particles):
        """Add dust particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(1, 2)
            
            # Dust colors (light gray/brown)
            base_color = random.randint(180, 220)
            color = (base_color - random.randint(0, 30), base_color, base_color + random.randint(0, 20))
            
            cv2.circle(img, (x, y), size, color, -1)
        
        return img

    def _add_confetti_particles(self, img, width, height, num_particles):
        """Add confetti particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            
            # Bright confetti colors
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            
            # Random confetti shapes
            shape = random.choice(['rectangle', 'circle', 'triangle'])
            size = random.randint(3, 8)
            
            if shape == 'rectangle':
                cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
            elif shape == 'circle':
                cv2.circle(img, (x, y), size//2, color, -1)
            elif shape == 'triangle':
                points = np.array([[x, y-size//2], [x-size//2, y+size//2], [x+size//2, y+size//2]], np.int32)
                cv2.fillPoly(img, [points], color)
        
        return img

    def _add_leaf_particles(self, img, width, height, num_particles):
        """Add falling leaf particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(4, 10)
            
            # Autumn leaf colors
            leaf_colors = [(139, 69, 19), (255, 140, 0), (255, 215, 0), (255, 69, 0), (34, 139, 34)]
            color = random.choice(leaf_colors)
            
            # Simple leaf shape (ellipse with stem)
            cv2.ellipse(img, (x, y), (size, size//2), random.randint(0, 180), 0, 360, color, -1)
            cv2.line(img, (x, y), (x, y+size//2), (101, 67, 33), 1)
        
        return img

    def _add_petal_particles(self, img, width, height, num_particles):
        """Add flower petal particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(3, 8)
            
            # Petal colors (pinks, whites, light colors)
            petal_colors = [(255, 182, 193), (255, 192, 203), (255, 240, 245), 
                           (240, 128, 128), (255, 228, 225), (255, 255, 240)]
            color = random.choice(petal_colors)
            
            # Petal shape (elongated ellipse)
            cv2.ellipse(img, (x, y), (size//2, size), random.randint(0, 360), 0, 360, color, -1)
        
        return img

    def _add_ember_particles(self, img, width, height, num_particles):
        """Add glowing ember particles"""
        overlay = img.copy()
        
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            
            # Hot ember colors
            ember_colors = [(255, 69, 0), (255, 140, 0), (255, 215, 0), (255, 99, 71)]
            color = random.choice(ember_colors)
            
            # Glow effect
            for radius in range(6, 1, -1):
                intensity = 1.0 - (6 - radius) / 5.0
                glow_color = tuple(int(c * intensity) for c in color)
                cv2.circle(overlay, (x, y), radius, glow_color, -1)
        
        alpha = 0.6
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_star_particles(self, img, width, height, num_particles):
        """Add star particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(2, 6)
            
            # Star colors (bright and twinkling)
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            
            # 4-point star
            cv2.line(img, (x-size, y), (x+size, y), color, 2)
            cv2.line(img, (x, y-size), (x, y+size), color, 2)
            cv2.line(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, 1)
            cv2.line(img, (x-size//2, y+size//2), (x+size//2, y-size//2), color, 1)
        
        return img

    def _add_crystal_particles(self, img, width, height, num_particles):
        """Add crystal particles"""
        for _ in range(num_particles):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(4, 12)
            
            # Crystal colors (cool and refractive)
            crystal_colors = [(173, 216, 230), (175, 238, 238), (224, 255, 255), 
                             (230, 230, 250), (248, 248, 255)]
            color = random.choice(crystal_colors)
            
            # Diamond/crystal shape
            points = np.array([
                [x, y-size], [x+size//2, y-size//2], [x+size, y],
                [x+size//2, y+size//2], [x, y+size], [x-size//2, y+size//2],
                [x-size, y], [x-size//2, y-size//2]
            ], np.int32)
            cv2.fillPoly(img, [points], color)
        
        return img

    def add_advanced_blur_effects(self, img: np.ndarray) -> np.ndarray:
        """Add various blur effects"""
        result = img.copy()
        blur_type = random.choice(self.blur_types)
        
        if blur_type == 'gaussian':
            result = self._add_gaussian_blur(result)
        elif blur_type == 'motion':
            result = self._add_motion_blur(result)
        elif blur_type == 'radial':
            result = self._add_radial_blur(result)
        elif blur_type == 'zoom':
            result = self._add_zoom_blur(result)
        elif blur_type == 'selective':
            result = self._add_selective_blur(result)
        elif blur_type == 'bokeh':
            result = self._add_bokeh_effect(result)
        
        return result

    def _add_gaussian_blur(self, img):
        """Add gaussian blur to random regions"""
        result = img.copy()
        height, width = result.shape[:2]
        
        # Create random regions to blur
        num_regions = random.randint(2, 4)
        
        for _ in range(num_regions):
            x1 = random.randint(0, width//2)
            y1 = random.randint(0, height//2)
            x2 = random.randint(width//2, width)
            y2 = random.randint(height//2, height)
            
            region = result[y1:y2, x1:x2].copy()
            kernel_size = random.choice([5, 7, 9, 11])
            blurred_region = cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
            
            # Blend blurred region back
            alpha = random.uniform(0.5, 0.8)
            result[y1:y2, x1:x2] = cv2.addWeighted(region, 1-alpha, blurred_region, alpha, 0)
        
        return result

    def _add_motion_blur(self, img):
        """Add motion blur effect"""
        result = img.copy()
        
        # Create motion blur kernel
        length = random.randint(10, 25)
        angle = random.randint(0, 180)
        
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        kernel[int((length-1)/2), :] = np.ones(length)
        kernel = kernel / length
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        
        # Apply motion blur to random region
        height, width = result.shape[:2]
        x1 = random.randint(0, width//3)
        y1 = random.randint(0, height//3)
        x2 = random.randint(2*width//3, width)
        y2 = random.randint(2*height//3, height)
        
        region = result[y1:y2, x1:x2].copy()
        blurred_region = cv2.filter2D(region, -1, kernel)
        
        alpha = random.uniform(0.6, 0.9)
        result[y1:y2, x1:x2] = cv2.addWeighted(region, 1-alpha, blurred_region, alpha, 0)
        
        return result

    def _add_radial_blur(self, img):
        """Add radial blur effect"""
        result = img.copy().astype(np.float32)
        height, width = result.shape[:2]
        
        center_x = random.randint(width//3, 2*width//3)
        center_y = random.randint(height//3, 2*height//3)
        
        # Create radial blur
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(width**2 + height**2) / 4
        
        blur_strength = np.minimum(distances / max_distance, 1.0) * 15
        
        for i in range(height):
            for j in range(width):
                blur_amount = int(blur_strength[i, j])
                if blur_amount > 1:
                    # Sample pixels in a circle around current pixel
                    samples = []
                    for angle in np.linspace(0, 2*np.pi, 8):
                        sample_x = int(j + blur_amount * np.cos(angle))
                        sample_y = int(i + blur_amount * np.sin(angle))
                        if 0 <= sample_x < width and 0 <= sample_y < height:
                            samples.append(result[sample_y, sample_x])
                    
                    if samples:
                        result[i, j] = np.mean(samples, axis=0)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_zoom_blur(self, img):
        """Add zoom blur effect"""
        result = img.copy()
        height, width = result.shape[:2]
        
        center_x = width // 2
        center_y = height // 2
        
        # Create zoom blur by blending multiple scaled versions
        zoom_layers = []
        for scale in np.linspace(1.0, 1.2, 5):
            scaled = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Center the scaled image
            scaled_h, scaled_w = scaled.shape[:2]
            start_x = (scaled_w - width) // 2
            start_y = (scaled_h - height) // 2
            
            if start_x >= 0 and start_y >= 0:
                cropped = scaled[start_y:start_y+height, start_x:start_x+width]
                if cropped.shape[:2] == (height, width):
                    zoom_layers.append(cropped.astype(np.float32))
        
        if zoom_layers:
            result = np.mean(zoom_layers, axis=0).astype(np.uint8)
        
        return result

    def _add_selective_blur(self, img):
        """Add selective focus blur"""
        result = img.copy()
        height, width = result.shape[:2]
        
        # Create focus mask (sharp in center, blurred at edges)
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width//2, height//2
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = min(width, height) // 3
        
        focus_mask = np.maximum(0, 1 - distances / max_distance)
        
        # Apply varying blur based on mask
        blurred = cv2.GaussianBlur(result, (15, 15), 0)
        
        for i in range(height):
            for j in range(width):
                focus_strength = focus_mask[i, j]
                result[i, j] = (focus_strength * result[i, j] + 
                               (1 - focus_strength) * blurred[i, j]).astype(np.uint8)
        
        return result

    def _add_bokeh_effect(self, img):
        """Add bokeh blur effect"""
        result = img.copy()
        overlay = np.zeros_like(result)
        height, width = result.shape[:2]
        
        # Add bokeh circles
        num_bokeh = random.randint(8, 15)
        for _ in range(num_bokeh):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(10, 30)
            
            # Soft, bright bokeh colors
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            
            # Create soft bokeh circle
            for r in range(radius, 0, -2):
                intensity = (radius - r) / radius * 0.3
                bokeh_color = tuple(int(c * intensity) for c in color)
                cv2.circle(overlay, (x, y), r, bokeh_color, -1)
        
        # Blur the background
        background = cv2.GaussianBlur(result, (21, 21), 0)
        
        # Combine with bokeh overlay
        result = cv2.addWeighted(background, 0.7, overlay, 0.3, 0)
        
        return result

    def add_text_overlays(self, img: np.ndarray) -> np.ndarray:
        """Add text overlays with various styles"""
        result = img.copy()
        height, width = result.shape[:2]
        
        # Random text samples
        text_samples = [
            "SAMPLE", "TEXT", "OVERLAY", "AI", "GENERATED", "SYNTHETIC",
            "TEST", "DEMO", "ALPHA", "BETA", "VERSION", "DRAFT",
            "WATERMARK", "PREVIEW", "PROTOTYPE", "CONCEPT"
        ]
        
        text_style = random.choice(self.text_styles)
        text = random.choice(text_samples)
        
        if text_style == 'watermark':
            result = self._add_watermark_text(result, text, width, height)
        elif text_style == 'graffiti':
            result = self._add_graffiti_text(result, text, width, height)
        elif text_style == 'neon':
            result = self._add_neon_text(result, text, width, height)
        elif text_style == 'embossed':
            result = self._add_embossed_text(result, text, width, height)
        elif text_style == 'shadow':
            result = self._add_shadow_text(result, text, width, height)
        elif text_style == 'outline':
            result = self._add_outline_text(result, text, width, height)
        
        return result

    def _add_watermark_text(self, img, text, width, height):
        """Add watermark style text"""
        overlay = img.copy()
        
        # Semi-transparent text
        font_scale = random.uniform(0.8, 1.5)
        color = (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200))
        
        # Position text
        x = random.randint(width//6, 4*width//6)
        y = random.randint(height//3, 2*height//3)
        
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        alpha = random.uniform(0.3, 0.5)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_graffiti_text(self, img, text, width, height):
        """Add graffiti style text"""
        overlay = img.copy()
        
        # Bold, colorful text
        font_scale = random.uniform(1.0, 2.0)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        x = random.randint(width//8, 6*width//8)
        y = random.randint(height//4, 3*height//4)
        
        # Add outline first
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 6)
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
        
        alpha = random.uniform(0.6, 0.8)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_neon_text(self, img, text, width, height):
        """Add neon glow text"""
        overlay = np.zeros_like(img)
        
        font_scale = random.uniform(0.8, 1.5)
        neon_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        color = random.choice(neon_colors)
        
        x = random.randint(width//8, 6*width//8)
        y = random.randint(height//3, 2*height//3)
        
        # Create glow effect
        for thickness in [8, 6, 4, 2]:
            intensity = 1.0 - (thickness - 2) / 6.0
            glow_color = tuple(int(c * intensity) for c in color)
            cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, glow_color, thickness)
        
        alpha = random.uniform(0.5, 0.7)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_embossed_text(self, img, text, width, height):
        """Add embossed text effect"""
        overlay = img.copy()
        
        font_scale = random.uniform(0.8, 1.5)
        x = random.randint(width//8, 6*width//8)
        y = random.randint(height//3, 2*height//3)
        
        # Create embossed effect with highlights and shadows
        # Shadow
        cv2.putText(overlay, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), 2)
        # Highlight
        cv2.putText(overlay, text, (x-1, y-1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 2)
        # Main text
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 128, 128), 2)
        
        alpha = random.uniform(0.4, 0.6)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_shadow_text(self, img, text, width, height):
        """Add text with drop shadow"""
        overlay = img.copy()
        
        font_scale = random.uniform(0.8, 1.5)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        shadow_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        
        x = random.randint(width//8, 6*width//8)
        y = random.randint(height//3, 2*height//3)
        
        # Draw shadow first
        shadow_offset = random.randint(3, 6)
        cv2.putText(overlay, text, (x+shadow_offset, y+shadow_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, shadow_color, 3)
        # Draw main text
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        alpha = random.uniform(0.5, 0.7)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_outline_text(self, img, text, width, height):
        """Add text with outline"""
        overlay = img.copy()
        
        font_scale = random.uniform(0.8, 1.5)
        text_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        outline_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        
        x = random.randint(width//8, 6*width//8)
        y = random.randint(height//3, 2*height//3)
        
        # Draw outline
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, 5)
        # Draw main text
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        
        alpha = random.uniform(0.5, 0.7)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def add_pattern_overlays(self, img: np.ndarray) -> np.ndarray:
        """Add various pattern overlays"""
        result = img.copy()
        height, width = result.shape[:2]
        
        pattern_type = random.choice(self.pattern_types)
        
        if pattern_type == 'stripes':
            result = self._add_stripe_pattern(result, width, height)
        elif pattern_type == 'checkerboard':
            result = self._add_checkerboard_pattern(result, width, height)
        elif pattern_type == 'dots':
            result = self._add_dot_pattern(result, width, height)
        elif pattern_type == 'waves':
            result = self._add_wave_pattern(result, width, height)
        elif pattern_type == 'spirals':
            result = self._add_spiral_pattern(result, width, height)
        elif pattern_type == 'fractals':
            result = self._add_fractal_pattern(result, width, height)
        
        return result

    def _add_stripe_pattern(self, img, width, height):
        """Add stripe pattern overlay"""
        overlay = img.copy()
        stripe_width = random.randint(8, 20)
        angle = random.randint(0, 180)
        
        # Create stripe pattern
        pattern = np.zeros((height, width), dtype=np.uint8)
        for i in range(0, max(width, height), stripe_width * 2):
            if angle == 0:  # Horizontal stripes
                if i < height:
                    pattern[i:i+stripe_width, :] = 255
            elif angle == 90:  # Vertical stripes
                if i < width:
                    pattern[:, i:i+stripe_width] = 255
            else:  # Diagonal stripes
                # This is a simplified diagonal pattern
                for y in range(height):
                    for x in range(width):
                        if (x + y) // stripe_width % 2 == 0:
                            pattern[y, x] = 255
        
        # Apply pattern with color
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        for c in range(3):
            overlay[:, :, c] = np.where(pattern > 0, color[c], overlay[:, :, c])
        
        alpha = random.uniform(0.15, 0.3)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_checkerboard_pattern(self, img, width, height):
        """Add checkerboard pattern overlay"""
        overlay = img.copy()
        square_size = random.randint(12, 25)
        
        # Create checkerboard pattern
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
                    cv2.rectangle(overlay, (x, y), (x+square_size, y+square_size), color, -1)
        
        alpha = random.uniform(0.2, 0.35)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_dot_pattern(self, img, width, height):
        """Add dot pattern overlay"""
        overlay = img.copy()
        dot_spacing = random.randint(15, 30)
        dot_size = random.randint(2, 6)
        
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        for y in range(dot_spacing, height, dot_spacing):
            for x in range(dot_spacing, width, dot_spacing):
                cv2.circle(overlay, (x, y), dot_size, color, -1)
        
        alpha = random.uniform(0.2, 0.4)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_wave_pattern(self, img, width, height):
        """Add wave pattern overlay"""
        overlay = img.copy()
        
        # Create wave lines
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        frequency = random.uniform(0.01, 0.03)
        amplitude = random.randint(20, 40)
        
        for offset in range(0, height, 20):
            points = []
            for x in range(width):
                y = int(offset + amplitude * math.sin(x * frequency))
                if 0 <= y < height:
                    points.append((x, y))
            
            for i in range(len(points)-1):
                cv2.line(overlay, points[i], points[i+1], color, 2)
        
        alpha = random.uniform(0.25, 0.4)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_spiral_pattern(self, img, width, height):
        """Add spiral pattern overlay"""
        overlay = img.copy()
        
        center_x, center_y = width // 2, height // 2
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        # Create spiral
        points = []
        for i in range(0, 720, 2):  # Two full rotations
            angle = math.radians(i)
            radius = i * 0.2
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            if 0 <= x < width and 0 <= y < height:
                points.append((x, y))
        
        for i in range(len(points)-1):
            cv2.line(overlay, points[i], points[i+1], color, 3)
        
        alpha = random.uniform(0.3, 0.5)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def _add_fractal_pattern(self, img, width, height):
        """Add simple fractal pattern overlay"""
        overlay = img.copy()
        
        # Simple fractal-like pattern using recursive circles
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        def draw_fractal_circles(x, y, radius, depth):
            if depth <= 0 or radius < 5:
                return
            
            cv2.circle(overlay, (x, y), radius, color, 2)
            
            # Draw smaller circles around the current one
            for angle in [0, 60, 120, 180, 240, 300]:
                new_x = int(x + radius * 0.7 * math.cos(math.radians(angle)))
                new_y = int(y + radius * 0.7 * math.sin(math.radians(angle)))
                if 0 <= new_x < width and 0 <= new_y < height:
                    draw_fractal_circles(new_x, new_y, radius // 2, depth - 1)
        
        # Start fractal from center
        draw_fractal_circles(width // 2, height // 2, 60, 3)
        
        alpha = random.uniform(0.25, 0.4)
        return cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    def add_distortion_effects(self, img: np.ndarray) -> np.ndarray:
        """Add various distortion effects"""
        result = img.copy()
        height, width = result.shape[:2]
        
        distortion_type = random.choice(['wave', 'ripple', 'lens', 'perspective'])
        
        if distortion_type == 'wave':
            result = self._add_wave_distortion(result, width, height)
        elif distortion_type == 'ripple':
            result = self._add_ripple_distortion(result, width, height)
        elif distortion_type == 'lens':
            result = self._add_lens_distortion(result, width, height)
        elif distortion_type == 'perspective':
            result = self._add_perspective_distortion(result, width, height)
        
        return result

    def _add_wave_distortion(self, img, width, height):
        """Add wave distortion effect"""
        result = np.zeros_like(img)
        
        amplitude = random.randint(5, 15)
        frequency = random.uniform(0.02, 0.05)
        
        for y in range(height):
            for x in range(width):
                # Calculate wave offset
                offset = int(amplitude * math.sin(x * frequency))
                new_y = y + offset
                
                if 0 <= new_y < height:
                    result[y, x] = img[new_y, x]
                else:
                    result[y, x] = img[y, x]
        
        return result

    def _add_ripple_distortion(self, img, width, height):
        """Add ripple distortion effect"""
        result = np.zeros_like(img)
        
        center_x = random.randint(width//3, 2*width//3)
        center_y = random.randint(height//3, 2*height//3)
        amplitude = random.randint(8, 20)
        frequency = random.uniform(0.1, 0.2)
        
        for y in range(height):
            for x in range(width):
                # Calculate distance from center
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Calculate ripple offset
                offset = int(amplitude * math.sin(distance * frequency))
                
                # Apply offset
                new_x = max(0, min(width-1, x + offset))
                new_y = max(0, min(height-1, y + offset))
                
                result[y, x] = img[new_y, new_x]
        
        return result

    def _add_lens_distortion(self, img, width, height):
        """Add lens distortion effect"""
        result = np.zeros_like(img)
        
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2
        strength = random.uniform(0.3, 0.7)
        
        for y in range(height):
            for x in range(width):
                # Calculate polar coordinates
                dx = x - center_x
                dy = y - center_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < max_radius:
                    # Apply barrel distortion
                    factor = 1 + strength * (distance / max_radius)**2
                    new_x = int(center_x + dx / factor)
                    new_y = int(center_y + dy / factor)
                    
                    if 0 <= new_x < width and 0 <= new_y < height:
                        result[y, x] = img[new_y, new_x]
                    else:
                        result[y, x] = img[y, x]
                else:
                    result[y, x] = img[y, x]
        
        return result

    def _add_perspective_distortion(self, img, width, height):
        """Add perspective distortion effect"""
        # Define source and destination points for perspective transform
        offset = random.randint(10, 30)
        
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst_points = np.float32([
            [offset, offset], 
            [width-offset, offset*2], 
            [width-offset*2, height-offset], 
            [offset*2, height-offset]
        ])
        
        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(img, matrix, (width, height))
        
        return result

    def create_semi_synthetic_image(self, img: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Create semi-synthetic image by applying 3-5 random modifications"""
        result = img.copy()
        applied_effects = []
        
        # Available modification methods with their names
        modification_methods = [
            ('diverse_geometric_shapes', self.add_diverse_geometric_shapes),
            ('advanced_light_effects', self.add_advanced_light_effects),
            ('diverse_particle_systems', self.add_diverse_particle_systems),
            ('advanced_blur_effects', self.add_advanced_blur_effects),
            ('text_overlays', self.add_text_overlays),
            ('pattern_overlays', self.add_pattern_overlays),
            ('distortion_effects', self.add_distortion_effects),
            ('selective_style_transfer', self.apply_selective_style_transfer),
            ('texture_overlay', self.add_texture_overlay),
            ('partial_inpainting', self.apply_partial_inpainting_effect)
        ]
        
        # Randomly select 3-5 modifications
        num_modifications = random.randint(3, 5)
        selected_methods = random.sample(modification_methods, num_modifications)
        
        print(f"    Applying {num_modifications} modifications...")
        
        for method_name, method_func in selected_methods:
            try:
                print(f"      Applying {method_name}...")
                result = method_func(result)
                applied_effects.append(method_name)
                print(f"      ✓ {method_name} applied successfully")
            except Exception as e:
                print(f"      ✗ {method_name} failed: {str(e)}")
                continue
        
        return result, applied_effects

    def apply_selective_style_transfer(self, img: np.ndarray) -> np.ndarray:
        """Apply style modifications to specific regions while preserving overall structure"""
        result = img.copy().astype(np.float32)
        height, width = result.shape[:2]
        
        # Create random region masks
        num_regions = random.randint(2, 4)
        
        for _ in range(num_regions):
            # Define region
            x1 = random.randint(0, width//2)
            y1 = random.randint(0, height//2)
            x2 = random.randint(width//2, width)
            y2 = random.randint(height//2, height)
            
            region = result[y1:y2, x1:x2].copy()
            
            # Apply style modification to region
            style_type = random.choice(['warm', 'cool', 'saturated', 'desaturated', 'vintage', 'dramatic'])
            
            if style_type == 'warm':
                region[:, :, 0] = np.minimum(region[:, :, 0] * 1.2 + 20, 255)  # More red
                region[:, :, 1] = np.minimum(region[:, :, 1] * 1.1 + 10, 255)  # Slight green
            elif style_type == 'cool':
                region[:, :, 2] = np.minimum(region[:, :, 2] * 1.3 + 25, 255)  # More blue
                region[:, :, 1] = np.minimum(region[:, :, 1] * 1.1 + 10, 255)  # Slight green
            elif style_type == 'saturated':
                # Increase saturation
                hsv_region = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv_region[:, :, 1] = np.minimum(hsv_region[:, :, 1] * 1.4, 255)
                region = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2RGB).astype(np.float32)
            elif style_type == 'desaturated':
                # Decrease saturation
                hsv_region = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv_region[:, :, 1] = hsv_region[:, :, 1] * 0.6
                region = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2RGB).astype(np.float32)
            elif style_type == 'vintage':
                # Vintage sepia effect
                region = region * [0.393, 0.769, 0.189] + region * [0.349, 0.686, 0.168] + region * [0.272, 0.534, 0.131]
            elif style_type == 'dramatic':
                # High contrast dramatic look
                region = np.power(region / 255.0, 0.7) * 255
            
            # Blend region back into image
            alpha = 0.7  # Blend factor
            result[y1:y2, x1:x2] = alpha * region + (1-alpha) * result[y1:y2, x1:x2]
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def add_texture_overlay(self, img: np.ndarray) -> np.ndarray:
        """Add procedural texture overlays"""
        result = img.copy()
        height, width = result.shape[:2]
        
        texture_types = ['noise', 'grain', 'crosshatch', 'dots', 'fabric', 'wood', 'metal']
        texture_type = random.choice(texture_types)
        
        # Create texture overlay
        if texture_type == 'noise':
            noise = np.random.randint(-30, 31, (height, width, 3))
            texture = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            alpha = 0.3
        elif texture_type == 'grain':
            grain = np.random.normal(0, 15, (height, width, 3))
            texture = np.clip(result.astype(np.float32) + grain, 0, 255).astype(np.uint8)
            alpha = 0.4
        elif texture_type == 'crosshatch':
            texture = result.copy()
            spacing = 8
            color = (100, 100, 100)
            for i in range(0, height, spacing):
                cv2.line(texture, (0, i), (width, i), color, 1)
            for i in range(0, width, spacing):
                cv2.line(texture, (i, 0), (i, height), color, 1)
            alpha = 0.2
        elif texture_type == 'dots':
            texture = result.copy()
            spacing = 12
            for y in range(0, height, spacing):
                for x in range(0, width, spacing):
                    cv2.circle(texture, (x, y), 2, (120, 120, 120), -1)
            alpha = 0.25
        elif texture_type == 'fabric':
            # Simulate fabric texture
            texture = result.copy()
            for i in range(0, height, 4):
                color_offset = random.randint(-20, 20)
                texture[i:i+2, :] = np.clip(texture[i:i+2, :].astype(int) + color_offset, 0, 255)
            alpha = 0.3
        elif texture_type == 'wood':
            # Simulate wood grain
            texture = result.copy()
            for y in range(height):
                grain_strength = 20 * math.sin(y * 0.1) + random.randint(-10, 10)
                texture[y, :] = np.clip(texture[y, :].astype(int) + grain_strength, 0, 255)
            alpha = 0.25
        elif texture_type == 'metal':
            # Simulate metallic texture
            texture = result.copy()
            for x in range(0, width, 2):
                highlight = random.randint(-15, 15)
                texture[:, x] = np.clip(texture[:, x].astype(int) + highlight, 0, 255)
            alpha = 0.3
        
        # Blend texture with original
        if texture_type in ['crosshatch', 'dots']:
            mask = (texture != result).any(axis=2)
            result[mask] = cv2.addWeighted(result, 1-alpha, texture, alpha, 0)[mask]
        else:
            result = cv2.addWeighted(result, 1-alpha, texture, alpha, 0)
        
        return result

    def apply_partial_inpainting_effect(self, img: np.ndarray) -> np.ndarray:
        """Simulate partial inpainting by modifying small regions"""
        result = img.copy()
        height, width = result.shape[:2]
        
        # Create several small regions to "inpaint"
        num_regions = random.randint(3, 6)
        
        for _ in range(num_regions):
            # Small region size
            region_w = random.randint(width//15, width//8)
            region_h = random.randint(height//15, height//8)
            
            x = random.randint(0, width - region_w)
            y = random.randint(0, height - region_h)
            
            # Get surrounding pixel colors for realistic inpainting
            surround_colors = []
            border_size = 5
            
            # Sample colors around the region
            for dx in [-border_size, border_size]:
                for dy in [-border_size, border_size]:
                    sample_x = np.clip(x + dx, 0, width-1)
                    sample_y = np.clip(y + dy, 0, height-1)
                    surround_colors.append(result[sample_y, sample_x])
            
            if surround_colors:
                avg_color = np.mean(surround_colors, axis=0).astype(np.uint8)
                
                # Fill region with blended colors based on surrounding pixels
                for ry in range(region_h):
                    for rx in range(region_w):
                        # Create gradient effect
                        center_x, center_y = region_w // 2, region_h // 2
                        dist_from_center = np.sqrt((rx - center_x)**2 + (ry - center_y)**2)
                        max_dist = np.sqrt(center_x**2 + center_y**2)
                        
                        # Blend between original and average color based on distance
                        blend_factor = min(dist_from_center / max_dist, 1.0) * 0.7
                        
                        original_pixel = result[y + ry, x + rx]
                        new_pixel = (blend_factor * avg_color + (1 - blend_factor) * original_pixel).astype(np.uint8)
                        
                        result[y + ry, x + rx] = new_pixel
        
        return result

    def process_tensor_batch_semi_synthetic(self, tensor_batch: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process batch to create semi-synthetic images with uint8 output"""
        batch_size = tensor_batch.shape[0]
        processed_images = []
        stats = {
            'total_input': batch_size,
            'successfully_processed': 0,
            'failed_processing': 0,
            'effects_applied': {},
            'average_modification_strength': 0.0
        }
        
        print(f"Creating diverse semi-synthetic images from batch of {batch_size} images...")
        
        modification_strengths = []
        
        for i in range(batch_size):
            try:
                original_img = self.tensor_to_numpy(tensor_batch[i])
                
                print(f"  Processing image {i+1}/{batch_size}")
                
                # Safety checks
                if np.all(original_img == 0) or original_img.std() < 5:
                    print(f"    ❌ Input image {i+1} invalid - skipping")
                    stats['failed_processing'] += 1
                    continue
                
                print(f"    Input OK - shape: {original_img.shape}, range: [{original_img.min()}, {original_img.max()}], std: {original_img.std():.2f}")
                
                # Create diverse semi-synthetic version
                semi_synthetic_img, applied_effects = self.create_semi_synthetic_image(original_img)
                
                # Safety check for output
                if np.all(semi_synthetic_img == 0):
                    print(f"    ❌ Output image {i+1} became black after processing - skipping")
                    stats['failed_processing'] += 1
                    continue
                
                print(f"    Output - shape: {semi_synthetic_img.shape}, range: [{semi_synthetic_img.min()}, {semi_synthetic_img.max()}], std: {semi_synthetic_img.std():.2f}")
                
                # Verify the modification occurred
                is_modified, difference = self.verify_transformation(original_img, semi_synthetic_img)
                
                if is_modified and semi_synthetic_img.std() > 1:
                    # Ensure output is exactly 224x224
                    if semi_synthetic_img.shape[:2] != (224, 224):
                        semi_synthetic_img = cv2.resize(semi_synthetic_img, (224, 224))
                    
                    # Convert to tensor with GUARANTEED uint8 format
                    processed_tensor = self.numpy_to_tensor(semi_synthetic_img)
                    processed_images.append(processed_tensor)
                    stats['successfully_processed'] += 1
                    modification_strengths.append(difference)
                    
                    # Track applied effects
                    for effect in applied_effects:
                        if effect in stats['effects_applied']:
                            stats['effects_applied'][effect] += 1
                        else:
                            stats['effects_applied'][effect] = 1
                    
                    print(f"    ✅ Image {i+1} successfully processed (effects: {', '.join(applied_effects)})")
                else:
                    stats['failed_processing'] += 1
                    print(f"    ❌ Image {i+1} skipped - insufficient modification or low variance")
                    
            except Exception as e:
                stats['failed_processing'] += 1
                print(f"    ❌ Image {i+1} error: {str(e)}")
                continue
        
        if len(processed_images) > 0:
            result_tensor = torch.stack(processed_images)
            stats['average_modification_strength'] = np.mean(modification_strengths)
            
            # Final verification of output format
            print(f"  FINAL OUTPUT VERIFICATION:")
            print(f"    Shape: {result_tensor.shape}")
            print(f"    Dtype: {result_tensor.dtype}")
            print(f"    Range: [{result_tensor.min().item()}, {result_tensor.max().item()}]")
            
            assert result_tensor.dtype == torch.uint8
            assert result_tensor.shape[1:] == (3, 224, 224)
            
        else:
            result_tensor = torch.empty((0, 3, 224, 224), dtype=torch.uint8)
            stats['average_modification_strength'] = 0.0
        
        print(f"\n  Result: {stats['successfully_processed']}/{batch_size} images successfully processed")
        print(f"  Average modification strength: {stats['average_modification_strength']:.2f}")
        
        return result_tensor, stats

    def process_directory_semi_synthetic(self, input_dir: str, output_dir: str, count: int, 
                                       batch_size: int = 1000) -> None:
        """Process directory to create diverse semi-synthetic images with uint8 output"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load mixed tensor and image files
        print(f"Loading {count} items from {input_dir}...")
        loaded_tensors, loaded_files = self.load_mixed_files(input_dir, count)
        
        # Concatenate all loaded tensors
        all_tensors = torch.cat(loaded_tensors, dim=0)[:count]
        print(f"Combined tensor shape: {all_tensors.shape}")
        
        # Process in batches
        total_input = 0
        total_successfully_processed = 0
        total_failed = 0
        batch_num = 0
        all_stats = []
        
        for start_idx in range(0, len(all_tensors), batch_size):
            end_idx = min(start_idx + batch_size, len(all_tensors))
            batch = all_tensors[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_num + 1} ({start_idx}-{end_idx-1})...")
            
            # Process batch to create diverse semi-synthetic images
            processed_batch, batch_stats = self.process_tensor_batch_semi_synthetic(batch)
            
            # Save batch if we have processed images
            if processed_batch.shape[0] > 0:
                output_filename = f"diverse_semi_synthetic_batch_{batch_num:04d}.pt"
                output_path = os.path.join(output_dir, output_filename)
                
                print(f"  Saving batch with shape: {processed_batch.shape}, dtype: {processed_batch.dtype}")
                torch.save(processed_batch, output_path)
                print(f"  Saved {processed_batch.shape[0]} diverse semi-synthetic images to {output_path}")
            else:
                print(f"  No images successfully processed in batch {batch_num + 1} - nothing saved")
            
            total_input += batch_stats['total_input']
            total_successfully_processed += batch_stats['successfully_processed']
            total_failed += batch_stats['failed_processing']
            all_stats.append(batch_stats)
            batch_num += 1
        
        # Print final summary
        print(f"\n" + "="*80)
        print("DIVERSE SEMI-SYNTHETIC PROCESSING COMPLETE")
        print("="*80)
        print(f"Total input images: {total_input}")
        print(f"Successfully processed: {total_successfully_processed}")
        print(f"Failed processing (skipped): {total_failed}")
        print(f"Success rate: {100*total_successfully_processed/total_input:.1f}%")
        print(f"Output batches created: {batch_num}")
        print(f"Output directory: {output_dir}")
        print(f"Output format: torch.uint8 tensors with shape (N, 3, 224, 224) and range [0, 255]")
        
        # Aggregate effect statistics
        all_effects = {}
        for stats in all_stats:
            for effect, count_val in stats['effects_applied'].items():
                if effect in all_effects:
                    all_effects[effect] += count_val
                else:
                    all_effects[effect] = count_val
        
        print("\nDiverse effects applied across all images:")
        for effect, count_val in sorted(all_effects.items()):
            print(f"  {effect}: {count_val} times")
        
        # Save comprehensive processing summary
        summary = {
            'input_directory': input_dir,
            'output_directory': output_dir,
            'processing_type': 'diverse_semi_synthetic',
            'output_format': {
                'tensor_dtype': 'torch.uint8',
                'tensor_shape': '(N, 3, 224, 224)',
                'value_range': '[0, 255]',
                'color_format': 'RGB'
            },
            'processing_summary': {
                'total_input_images': total_input,
                'successfully_processed': total_successfully_processed,
                'failed_processing': total_failed,
                'success_rate_percentage': 100*total_successfully_processed/total_input if total_input > 0 else 0,
                'output_batches_created': batch_num,
                'modifications_per_image': '3-5 random effects per image'
            },
            'available_modification_categories': {
                'diverse_geometric_shapes': [
                    'circles', 'rectangles', 'triangles', 'stars', 'polygons',
                    'ellipses', 'diamonds', 'hexagons', 'crescents', 'arrows',
                    'hearts', 'crosses', 'spirals', 'bezier_curves', 'wave_lines'
                ],
                'advanced_light_effects': [
                    'lens_flare', 'light_rays', 'spotlight', 'aurora', 'lightning',
                    'rainbow', 'sun_rays', 'laser_beams', 'glow_effects', 'halo'
                ],
                'diverse_particle_systems': [
                    'sparkles', 'bubbles', 'snow', 'rain', 'fireflies', 'dust',
                    'confetti', 'leaves', 'petals', 'embers', 'stars', 'crystals'
                ],
                'advanced_blur_effects': [
                    'gaussian', 'motion', 'radial', 'zoom', 'selective', 'bokeh'
                ],
                'text_overlays': [
                    'watermark', 'graffiti', 'neon', 'embossed', 'shadow', 'outline'
                ],
                'pattern_overlays': [
                    'stripes', 'checkerboard', 'dots', 'waves', 'spirals', 'fractals'
                ],
                'distortion_effects': [
                    'wave', 'ripple', 'lens', 'perspective'
                ],
                'style_transfer': [
                    'warm', 'cool', 'saturated', 'desaturated', 'vintage', 'dramatic'
                ],
                'texture_overlays': [
                    'noise', 'grain', 'crosshatch', 'dots', 'fabric', 'wood', 'metal'
                ],
                'partial_inpainting': [
                    'smart_region_filling_based_on_surrounding_content'
                ]
            },
            'quality_features': {
                'preserves_original_content': True,
                'applies_3_to_5_modifications_per_image': True,
                'highly_diverse_realistic_modifications': True,
                'multiple_blending_modes': True,
                'advanced_visual_effects': True,
                'maintains_image_structure': True,
                'professional_quality_overlays': True
            },
            'effects_statistics': all_effects,
            'batch_statistics': all_stats,
            'source_files': loaded_files
        }
        
        summary_path = os.path.join(output_dir, "diverse_semi_synthetic_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nDetailed processing summary saved to: {summary_path}")


def main():
    """Example usage for creating diverse semi-synthetic images with uint8 output"""
    
    # Initialize processor
    processor = EnhancedSemiSyntheticProcessor()
    
    # Example usage - replace with your actual paths
    input_directory = "/mnt/data/image-datasets-train/train/real"  # Replace with your input directory
    output_directory = "semi_synthetic_results"  # Replace with your output directory
    image_count = 100  # Number of images to process
    batch_size = 5000   # Size of output .pt files
    
    try:
        processor.process_directory_semi_synthetic(
            input_dir=input_directory,
            output_dir=output_directory,
            count=image_count,
            batch_size=batch_size
        )
        
        print("\n" + "="*90)
        print("DIVERSE SEMI-SYNTHETIC MODIFICATION METHODS SUCCESSFULLY APPLIED:")
        print("="*90)
        
        print("✓ DIVERSE GEOMETRIC SHAPES (15 TYPES):")
        print("  • Basic: circles, rectangles, triangles, ellipses")
        print("  • Advanced: stars, polygons, diamonds, hexagons, crescents")
        print("  • Artistic: arrows, hearts, crosses, spirals, bezier_curves, wave_lines")
        print("  • Multiple blending modes: normal, multiply, screen, overlay")
        print("  • Gradient and solid color fills")
        print("")
        
        print("✓ ADVANCED LIGHT EFFECTS (10 TYPES):")
        print("  • Realistic: lens_flare, light_rays, spotlight, sun_rays")
        print("  • Dramatic: aurora, lightning, rainbow, laser_beams")
        print("  • Atmospheric: glow_effects, halo")
        print("  • Professional lighting simulation")
        print("")
        
        print("✓ DIVERSE PARTICLE SYSTEMS (12 TYPES):")
        print("  • Natural: snow, rain, dust, leaves, petals")
        print("  • Magical: sparkles, fireflies, stars, crystals")
        print("  • Festive: bubbles, confetti, embers")
        print("  • Realistic physics and lighting")
        print("")
        
        print("✓ ADVANCED BLUR EFFECTS (6 TYPES):")
        print("  • gaussian, motion, radial, zoom, selective, bokeh")
        print("  • Professional depth-of-field simulation")
        print("  • Artistic focus effects")
        print("")
        
        print("✓ TEXT OVERLAYS (6 STYLES):")
        print("  • watermark, graffiti, neon, embossed, shadow, outline")
        print("  • Random text from realistic samples")
        print("  • Professional typography effects")
        print("")
        
        print("✓ PATTERN OVERLAYS (6 TYPES):")
        print("  • stripes, checkerboard, dots, waves, spirals, fractals")
        print("  • Procedural pattern generation")
        print("  • Semi-transparent blending")
        print("")
        
        print("✓ DISTORTION EFFECTS (4 TYPES):")
        print("  • wave, ripple, lens, perspective")
        print("  • Realistic optical distortions")
        print("  • Maintains image integrity")
        print("")
        
        print("✓ ENHANCED STYLE TRANSFER (6 STYLES):")
        print("  • warm, cool, saturated, desaturated, vintage, dramatic")
        print("  • Regional selective application")
        print("  • Professional color grading")
        print("")
        
        print("✓ EXPANDED TEXTURE OVERLAYS (7 TYPES):")
        print("  • Digital: noise, grain, crosshatch, dots")
        print("  • Natural: fabric, wood, metal")
        print("  • Realistic material simulation")
        print("")
        
        print("✓ SMART PARTIAL INPAINTING:")
        print("  • Context-aware region filling")
        print("  • Surrounding color analysis")
        print("  • Gradient blending for natural appearance")
        print("")
        
        print("✓ GUARANTEED OUTPUT FORMAT:")
        print("  • Tensor dtype: torch.uint8")
        print("  • Tensor shape: (N, 3, 224, 224)")
        print("  • Value range: [0, 255]")
        print("  • Color format: RGB")
        print("  • 3-5 random modifications per image")
        print("  • Maximum diversity and realism")
        print("  • Professional quality visual effects")
        print("="*90)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure to:")
        print("1. Replace input directory path with your actual directory")
        print("2. Ensure input directory contains .pt, .pth or image files")
        print("3. Input tensors should have shape [N, 3, 224, 224] or [N, 224, 224, 3]")
        print("4. Install required packages: torch, opencv-python, pillow, scikit-learn, numpy")


def test_all_modifications():
    """Test all modification methods on sample images"""
    processor = EnhancedSemiSyntheticProcessor()
    
    # Create test images
    test_images = []
    
    # Test image 1: Colorful scene
    img1 = np.zeros((224, 224, 3), dtype=np.uint8)
    img1[:100, :] = [135, 206, 235]  # Sky
    img1[100:, :] = [34, 139, 34]   # Ground
    cv2.circle(img1, (112, 50), 25, (255, 255, 0), -1)  # Sun
    test_images.append(("landscape", img1))
    
    # Test image 2: Portrait-like
    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img2[:] = [200, 180, 160]  # Background
    cv2.circle(img2, (112, 100), 60, (220, 200, 180), -1)  # Face
    test_images.append(("portrait", img2))
    
    print("Testing all diverse modification methods...")
    
    modification_categories = [
        ("Diverse Geometric Shapes", processor.add_diverse_geometric_shapes),
        ("Advanced Light Effects", processor.add_advanced_light_effects),
        ("Diverse Particle Systems", processor.add_diverse_particle_systems),
        ("Advanced Blur Effects", processor.add_advanced_blur_effects),
        ("Text Overlays", processor.add_text_overlays),
        ("Pattern Overlays", processor.add_pattern_overlays),
        ("Distortion Effects", processor.add_distortion_effects),
        ("Enhanced Style Transfer", processor.apply_selective_style_transfer),
        ("Expanded Texture Overlays", processor.add_texture_overlay),
        ("Smart Partial Inpainting", processor.apply_partial_inpainting_effect)
    ]
    
    for img_name, test_img in test_images:
        print(f"\n--- Testing on {img_name} image ---")
        
        for category_name, method_func in modification_categories:
            try:
                modified = method_func(test_img.copy())
                is_modified, difference = processor.verify_transformation(test_img, modified)
                
                status = "✅ SUCCESS" if is_modified else "⚠️  SUBTLE"
                print(f"{category_name:25} - {status} - Difference: {difference:8.2f}")
                
            except Exception as e:
                print(f"{category_name:25} - ❌ ERROR: {str(e)}")
    
    # Test full pipeline
    print(f"\n--- Testing full diverse pipeline ---")
    for img_name, test_img in test_images:
        try:
            semi_synthetic, applied_effects = processor.create_semi_synthetic_image(test_img)
            is_modified, difference = processor.verify_transformation(test_img, semi_synthetic)
            
            print(f"{img_name} pipeline - Modified: {is_modified} - Difference: {difference:.2f}")
            print(f"  Applied effects: {', '.join(applied_effects)}")
            
            # Test tensor conversion
            tensor_output = processor.numpy_to_tensor(semi_synthetic)
            print(f"  Tensor output: shape={tensor_output.shape}, dtype={tensor_output.dtype}, range=[{tensor_output.min().item()}, {tensor_output.max().item()}]")
            
        except Exception as e:
            print(f"{img_name} pipeline - ERROR: {str(e)}")


def create_diverse_sample_showcase():
    """Create a showcase of all diverse effects"""
    processor = EnhancedSemiSyntheticProcessor()
    
    # Create sample directory
    sample_dir = "diverse_effects_showcase"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Base test image
    base_img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Create a more interesting base image
    base_img[:112, :] = [100, 150, 200]  # Sky blue top
    base_img[112:, :] = [150, 100, 80]   # Brown bottom
    cv2.circle(base_img, (56, 56), 30, (255, 255, 100), -1)  # Sun
    cv2.rectangle(base_img, (168, 140), (200, 200), (80, 60, 40), -1)  # Building
    
    print("Creating diverse effects showcase...")
    
    # Save original
    cv2.imwrite(os.path.join(sample_dir, "00_original.png"), 
               cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR))
    
    # Test each category multiple times to show variety
    all_methods = [
        ("geometric_shapes", processor.add_diverse_geometric_shapes),
        ("light_effects", processor.add_advanced_light_effects),
        ("particles", processor.add_diverse_particle_systems),
        ("blur_effects", processor.add_advanced_blur_effects),
        ("text_overlays", processor.add_text_overlays),
        ("patterns", processor.add_pattern_overlays),
        ("distortions", processor.add_distortion_effects),
        ("style_transfer", processor.apply_selective_style_transfer),
        ("textures", processor.add_texture_overlay),
        ("inpainting", processor.apply_partial_inpainting_effect)
    ]
    
    sample_count = 0
    for method_name, method_func in all_methods:
        # Create 3 variations of each effect
        for variation in range(3):
            try:
                modified = method_func(base_img.copy())
                filename = f"{sample_count:02d}_{method_name}_v{variation+1}.png"
                filepath = os.path.join(sample_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(modified, cv2.COLOR_RGB2BGR))
                print(f"  Created: {filename}")
                sample_count += 1
            except Exception as e:
                print(f"  Error creating {method_name} v{variation+1}: {str(e)}")
    
    # Create full pipeline examples
    print("\nCreating full pipeline examples...")
    for i in range(5):
        try:
            semi_synthetic, effects = processor.create_semi_synthetic_image(base_img.copy())
            filename = f"pipeline_example_{i+1:02d}.png"
            filepath = os.path.join(sample_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(semi_synthetic, cv2.COLOR_RGB2BGR))
            print(f"  Created: {filename} (effects: {', '.join(effects)})")
        except Exception as e:
            print(f"  Error creating pipeline example {i+1}: {str(e)}")
    
    print(f"\nDiverse effects showcase created in: {sample_dir}/")
    print(f"Total samples created: {sample_count + 5}")


if __name__ == "__main__":
    # Run main processing
    main()
    
    print("\n" + "="*90)
    print("ADDITIONAL TESTING AND SHOWCASE OPTIONS:")
    print("="*90)
    print("Uncomment the lines below to run additional tests:")
    print("")
    print("# Test all modification methods individually:")
    print("# test_all_modifications()")
    print("")
    print("# Create a showcase of all diverse effects:")
    print("# create_diverse_sample_showcase()")
    
    # Uncomment to test all individual modifications
    # test_all_modifications()
    
    # Uncomment to create visual showcase of all effects
    # create_diverse_sample_showcase()