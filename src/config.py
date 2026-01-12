AWS_BUCKET = "ai-people-reader-storage"

# Full S3 prefixes (bucket + folder)
PENDING_PREFIX = f"{AWS_BUCKET}/jobs/pending/"
PROCESSING_PREFIX = f"{AWS_BUCKET}/jobs/processing/"
FAILED_PREFIX = f"{AWS_BUCKET}/jobs/failed/"
OUTPUT_PREFIX = f"{AWS_BUCKET}/jobs/output/"

# Region
REGION_NAME = "ap-southeast-1"

