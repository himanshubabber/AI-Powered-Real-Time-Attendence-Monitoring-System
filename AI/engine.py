"""
================================================================================
   DEEP LEARNING BIOMETRIC ENGINE (ENTERPRISE EDITION)
================================================================================
   Module:      engine.py
   Author:      Himanshu & Team
   Version:     3.5.0 (Production)
   Description: 
        The core neural processing unit for the Attendance System. 
        This engine manages the full lifecycle of facial biometric processing:
        
        1. Image Ingestion & Preprocessing (RGB normalization)
        2. Face Detection (MTCNN Multi-Stage Cascade)
        3. Feature Extraction (Inception-ResNet-v1 Deep CNN)
        4. Vector Math (Euclidean L2 Distance Matching)

   Dependencies: PyTorch, FaceNet-PyTorch, PIL, NumPy
================================================================================
"""

import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import io
import time
import logging
import sys
import warnings

# Suppress internal PyTorch warnings for cleaner logs
warnings.filterwarnings("ignore")

# --- 1. LOGGING CONFIGURATION ---
# Sets up a professional logging format: [TIME] [LEVEL] [MESSAGE]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AI_Brain")

# --- 2. ENGINE CONFIGURATION ---
class EngineConfig:
    """Centralized Configuration for AI Hyperparameters"""
    # Hardware
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Face Detection (MTCNN)
    IMG_SIZE = 160
    MARGIN = 20
    MIN_FACE_SIZE = 40      # Ignore faces smaller than 40px
    THRESHOLDS = [0.6, 0.7, 0.7] # Confidence thresholds for P-Net, R-Net, O-Net
    
    # Face Recognition (ResNet)
    PRETRAINED_WEIGHTS = 'vggface2' # Trained on 3.3M faces
    MATCH_THRESHOLD = 0.75          # Distance < 0.75 implies same person (Stricter than 0.8)

# --- 3. CORE INTELLIGENCE CLASS ---
class FaceEngine:
    """
    The main AI controller. Instantiating this class loads the heavy Deep Learning
    models into VRAM (GPU) or RAM (CPU) to ensure low-latency inference.
    """

    def __init__(self):
        logger.info("="*50)
        logger.info(f"🚀 INITIALIZING BIOMETRIC ENGINE ON: {EngineConfig.DEVICE}")
        logger.info("="*50)
        
        self.device = EngineConfig.DEVICE
        
        try:
            # ---------------------------------------------------------
            # LAYER 1: FACE DETECTION (MTCNN)
            # ---------------------------------------------------------
            # We initialize TWO detectors:
            # 1. 'single': Optimized for Registration (finds largest face only)
            # 2. 'group':  Optimized for Attendance (finds ALL faces)
            logger.info("1. Loading MTCNN Face Detection Networks...")
            
            self.mtcnn_single = MTCNN(
                image_size=EngineConfig.IMG_SIZE,
                margin=EngineConfig.MARGIN,
                keep_all=False,
                select_largest=True, # Critical for registration selfies
                thresholds=EngineConfig.THRESHOLDS,
                device=self.device
            )
            
            self.mtcnn_group = MTCNN(
                image_size=EngineConfig.IMG_SIZE, 
                margin=EngineConfig.MARGIN, 
                keep_all=True,       # Detect multiple people
                min_face_size=EngineConfig.MIN_FACE_SIZE,
                thresholds=EngineConfig.THRESHOLDS,
                device=self.device
            )

            # ---------------------------------------------------------
            # LAYER 2: FEATURE EXTRACTION (Inception-ResNet-v1)
            # ---------------------------------------------------------
            logger.info(f"2. Loading Inception-ResNet-v1 (Weights: {EngineConfig.PRETRAINED_WEIGHTS})...")
            
            self.resnet = InceptionResnetV1(
                pretrained=EngineConfig.PRETRAINED_WEIGHTS
            ).eval().to(self.device)
            
            # Count Parameters for visual impact
            params = sum(p.numel() for p in self.resnet.parameters())
            logger.info(f"   -> Deep Neural Network Loaded. Parameters: {params:,}")
            logger.info("✅ SYSTEM READY.")
            
        except Exception as e:
            logger.critical(f"❌ CRITICAL FAILURE: Model Loading Error -> {e}")
            raise e

    def inspect_architecture(self):
        """
        Debug method to print the internal layers of the ResNet model.
        Useful for project reports to demonstrate 'Deep Learning' depth.
        """
        print("\n--- NEURAL NETWORK ARCHITECTURE ---")
        for name, module in self.resnet.named_children():
            if "Block" in name or "Mixed" in name:
                print(f"[Layer] {name:<15} : {module.__class__.__name__}")
        print("-----------------------------------\n")

    def _preprocess_image(self, image_bytes):
        """
        Internal pipeline to sanitize input images.
        1. Decode Bytes -> PIL Image
        2. Convert Color Space (RGBA/Grayscale -> RGB)
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            
            # Deep Learning models expect 3 Channels (Red, Green, Blue).
            # If we get a PNG (4 channels) or Black/White (1 channel), it crashes.
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Image Decoding Failed: {e}")
            return None

    def get_single_embedding(self, image_bytes):
        """
        [REGISTRATION MODE]
        Extracts a 512-D vector from a single user photo.
        Returns: List[float] or None
        """
        start_time = time.time()
        img = self._preprocess_image(image_bytes)
        
        if img is None: 
            return None

        # Step A: Detection
        # Returns cropped tensor shape: (3, 160, 160)
        img_cropped = self.mtcnn_single(img)
        
        if img_cropped is None:
            logger.warning("Registration Failed: No face detected.")
            return None
            
        # Step B: Recognition
        # Unsqueeze adds batch dimension: (1, 3, 160, 160)
        try:
            img_embedding = self.resnet(img_cropped.to(self.device).unsqueeze(0))
        except RuntimeError as e:
            logger.error(f"Inference Error: {e}")
            return None
        
        # Step C: Serialization
        # Detach from GPU graph -> Move to CPU -> Convert to standard List
        vector = img_embedding.detach().cpu().numpy().tolist()[0]
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"⚡ Vector Generated in {duration:.2f}ms. Vector Length: {len(vector)}")
        
        return vector

    def recognize_faces_in_group(self, image_bytes, known_students):
        """
        [ATTENDANCE MODE]
        1. Detects ALL faces in a group photo.
        2. Compares each face against the 'known_students' database.
        
        Args:
            image_bytes: Raw photo data
            known_students: List of dicts [{'roll': '101', 'vector': [...]}]
        
        Returns:
            List[str]: List of Roll Numbers marked present.
        """
        overall_start = time.time()
        img = self._preprocess_image(image_bytes)
        if img is None: return []

        # --- PHASE 1: DETECTION ---
        # Returns a stack of tensors: (N_Faces, 3, 160, 160)
        faces_aligned = self.mtcnn_group(img)
        
        if faces_aligned is None:
            logger.info("Analysis Complete: 0 Faces found.")
            return []

        num_faces = len(faces_aligned)
        logger.info(f"📸 Group Photo Analysis: Found {num_faces} Faces.")

        # --- PHASE 2: BATCH INFERENCE ---
        # We pass ALL faces to the GPU at once (Batch Processing) for speed
        with torch.no_grad():
            group_embeddings = self.resnet(faces_aligned.to(self.device)).detach().cpu()

        # --- PHASE 3: VECTOR MATCHING ---
        if not known_students:
            logger.warning("No known student data provided for comparison.")
            return []

        # Convert Node.js JSON data into PyTorch Tensor Matrix for fast math
        # Shape: (Num_Students, 512)
        known_vectors = [s['vector'] for s in known_students]
        known_rolls = [s['roll'] for s in known_students]
        known_tensor = torch.tensor(known_vectors)

        present_roll_nos = []

        # Compare every detected face against the entire class list
        for i, face_emb in enumerate(group_embeddings):
            # Calculate Euclidean Distance (L2 Norm)
            # This line subtracts the current face from ALL students at once
            distances = (face_emb - known_tensor).norm(p=2, dim=1)
            
            # Find the minimum distance (closest match)
            min_dist, min_idx = distances.min(dim=0)
            min_dist_val = min_dist.item()
            
            status = "UNKNOWN"
            match_roll = "---"

            # Check if match is strong enough
            if min_dist_val < EngineConfig.MATCH_THRESHOLD:
                match_roll = known_rolls[min_idx.item()]
                present_roll_nos.append(match_roll)
                status = "MATCH"
            
            logger.info(f"   > Face {i+1}: Dist={min_dist_val:.4f} | Status={status} | ID={match_roll}")

        total_time = (time.time() - overall_start) * 1000
        logger.info(f"🏁 Attendance Process Complete. Time: {total_time:.2f}ms. Found: {len(set(present_roll_nos))}")

        return list(set(present_roll_nos))
