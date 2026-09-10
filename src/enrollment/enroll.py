"""
Runtime Few-Shot Keyword Enrollment Manager
SIH Problem Statement 26172 - Milestone 9

Simulates edge device enrollment:
1. Takes K spoken samples of an unseen keyword (e.g., 'ZORA')
2. Extracts MFCC features (98, 13)
3. Computes L2-normalized 32-D embeddings via frozen encoder
4. Computes prototype vector centroid: p = L2_norm(mean(e))
5. Exports C/C++ firmware header for ESP32-S3 deployment
"""

import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, r"D:\SIH_Model")
from src.features.mfcc import MFCCFeatureExtractor
from src.models.prototype import compute_prototype, compute_cosine_similarity

class KeywordEnrollmentManager:
    def __init__(self, encoder, feature_extractor=None):
        self.encoder = encoder
        self.feature_extractor = feature_extractor or MFCCFeatureExtractor(sample_rate=16000, n_mfcc=13, n_mels=40)

    def extract_embedding_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """Extracts L2-normalized 32-D embedding from 1D audio array."""
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        elif len(audio) > 16000:
            audio = audio[:16000]

        feat = self.feature_extractor.extract(audio)  # (98, 13)
        feat_batch = feat[np.newaxis, :, :, np.newaxis]
        emb = self.encoder(feat_batch, training=False).numpy()[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def extract_embedding_from_file(self, filepath: str) -> np.ndarray:
        """Loads WAV file and extracts L2-normalized embedding."""
        audio, sr = sf.read(filepath, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        return self.extract_embedding_from_audio(audio)

    def enroll(self, audio_inputs, keyword_name="ZORA") -> dict:
        """
        Enrolls target keyword using K spoken utterances.
        audio_inputs: List of filepaths or list of numpy audio arrays.
        """
        embeddings = []
        for inp in audio_inputs:
            if isinstance(inp, str):
                emb = self.extract_embedding_from_file(inp)
            elif isinstance(inp, np.ndarray):
                emb = self.extract_embedding_from_audio(inp)
            else:
                raise ValueError("Expected str filepath or np.ndarray audio")
            embeddings.append(emb)

        embeddings = np.array(embeddings, dtype=np.float32)
        k_shots = len(embeddings)

        # Compute centroid prototype
        prototype = compute_prototype(embeddings)

        # Compute intra-enrollment pairwise similarities
        if k_shots >= 2:
            sim_matrix = np.dot(embeddings, embeddings.T)
            # Upper triangle off-diagonal elements
            triu_idx = np.triu_indices(k_shots, k=1)
            pair_sims = sim_matrix[triu_idx]
            intra_mean = float(np.mean(pair_sims))
            intra_min = float(np.min(pair_sims))
            intra_std = float(np.std(pair_sims))
        else:
            intra_mean = 1.0
            intra_min = 1.0
            intra_std = 0.0

        # Compute distance of individual shots to centroid
        shot_to_proto_sims = [compute_cosine_similarity(prototype, emb) for emb in embeddings]

        return {
            "keyword_name": keyword_name.upper(),
            "k_shots": k_shots,
            "prototype": prototype,
            "embeddings": embeddings,
            "intra_similarity_mean": round(intra_mean, 4),
            "intra_similarity_min": round(intra_min, 4),
            "intra_similarity_std": round(intra_std, 4),
            "shot_to_prototype_sims": [round(float(s), 4) for s in shot_to_proto_sims]
        }

    def export_prototype_c_header(self, prototype: np.ndarray, keyword_name="ZORA",
                                  output_filepath=r"D:\SIH_Model\src\deployment\esp32\keyword_prototype.h"):
        """Exports the prototype array to a C header file for ESP32-S3 firmware."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        dim = len(prototype)
        proto_vals = ", ".join([f"{v:.7f}f" for v in prototype])

        header_content = f"""/*
 * Auto-generated Keyword Prototype Header for ESP32-S3
 * SIH Problem Statement 26172
 * Keyword: {keyword_name.upper()}
 * Embedding Dimension: {dim}
 * Hypersphere Normalized: True (||p||_2 = 1.0)
 */

#ifndef KEYWORD_PROTOTYPE_H_
#define KEYWORD_PROTOTYPE_H_

#define ENROLLED_KEYWORD_NAME "{keyword_name.upper()}"
#define EMBEDDING_DIMENSION {dim}

// L2-normalized prototype vector centroid
static const float ENROLLED_KEYWORD_PROTOTYPE[{dim}] = {{
    {proto_vals}
}};

#endif // KEYWORD_PROTOTYPE_H_
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(header_content)
        return output_filepath
