"""
Smart Reassort — Upload du pickle sur S3
==========================================
Upload de model.pkl vers s3://<bucket>/smart-reassort/models/model_latest.pkl

Configuration via .env à la racine du projet :
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_DEFAULT_REGION=eu-west-3
    S3_BUCKET=ton-bucket-name

Usage standalone :
    python newcode/train/save_pickle.py

Usage Airflow :
    from newcode.train.save_pickle import run_upload
    run_upload(model_path="...", bucket="...", s3_key="...")
"""

import os
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# ───────────────────────────────────────────────
# CHARGEMENT DU .env
# ───────────────────────────────────────────────
load_dotenv()

# ───────────────────────────────────────────────
# LOGGING
# ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
# CONSTANTES PAR DÉFAUT
# ───────────────────────────────────────────────
DEFAULT_MODEL_PATH = "newcode/train/model.pkl"
DEFAULT_S3_KEY     = "smart-reassort/models/model_latest.pkl"


# ═══════════════════════════════════════════════
# UPLOAD S3
# ═══════════════════════════════════════════════

def run_upload(
    model_path: str,
    bucket: str,
    s3_key: str = DEFAULT_S3_KEY,
) -> str:
    """
    Upload un fichier local vers S3.
    """
    local_path = Path(model_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Fichier local introuvable : {model_path}")
    
    size_mb = local_path.stat().st_size / 1024 / 1024
    logger.info(f"  Fichier local : {model_path} ({size_mb:.2f} Mo)")
    logger.info(f"  Destination   : s3://{bucket}/{s3_key}")
    
    s3 = boto3.client("s3")
    
    try:
        logger.info("  Upload en cours...")
        s3.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )
    except NoCredentialsError:
        logger.error("  ✗ Credentials AWS manquants (vérifier .env)")
        raise
    except ClientError as e:
        logger.error(f"  ✗ Erreur S3 : {e}")
        raise
    
    s3_url = f"s3://{bucket}/{s3_key}"
    logger.info(f"  ✓ Upload réussi : {s3_url}")
    return s3_url


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "Variable S3_BUCKET manquante dans .env. "
            "Crée un .env à la racine avec S3_BUCKET=ton-bucket"
        )
    
    run_upload(
        model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
        bucket=bucket,
        s3_key=os.getenv("S3_KEY", DEFAULT_S3_KEY),
    )