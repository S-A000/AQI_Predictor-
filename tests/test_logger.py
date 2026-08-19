from src.utils.logger import get_logger

logger = get_logger("pipeline")
logger.info("Pipeline Started")

api_logger = get_logger("api")
api_logger.info("Weather API Called")

logger.warning("Warning Test")

logger.error("Error Test")

print("Done")