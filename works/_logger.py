import os
import sys

from loguru import logger as loguru_logger

loguru_logger.configure(
    handlers=[
        dict(
            sink=sys.stderr,
            filter=lambda record: record["extra"]["console"],
            level="INFO",
        )
    ]
)

logger = loguru_logger.bind(console=True)
tracker_logger = loguru_logger.bind(console=False, tracker=True)


def add_file_handler_to_logger(
    name: str,
    dir_path: str = "",
    level="DEBUG",
):
    loguru_logger.add(
        sink=os.path.join(dir_path, f"{name}-{level}.log"),
        level=level,
        rotation="1 day",
        retention="7 days",
        filter=lambda record: "tracker" not in record["extra"]
        and "console" in record["extra"]
        and record["level"] != "ERROR"
        if level != "ERROR"
        else True,
        enqueue=True,
    )


def add_track_handler(dir_path: str = ""):
    tracker_logger.add(
        sink=os.path.join(dir_path, "tracker.log"),
        format="{message}",
        rotation="1 day",
        retention="7 days",
        filter=lambda record: "tracker" in record["extra"] and record["extra"]["tracker"],
        enqueue=True,
    )