#!/usr/bin/env python3
"""
通用工具模块 - PersoanlQuery 技能共享库

# 权限验证

提供以下功能：
- 配置加载和验证
- 日志初始化和管理
- JSON 文件操作（保存和加载）
- 文件和路径操作
- 进度追踪和统计
- 失败项记录

使用示例：
    from common_utils import (
        load_config, setup_logging, save_json, load_json,
        ensure_dir, get_timestamped_filename
    )

    # 初始化日志
    logger = setup_logging("my_script", "logs")

    # 加载配置
    config = load_config("config.json")

    # 保存数据
    save_json(data, "output.json")

    # 加载数据
    data = load_json("input.json")

    # 创建目录
    output_dir = ensure_dir("./results")
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


# ============================================================================
# 日志工具
# ============================================================================

def setup_logging(
    log_name: str = "pipeline",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    统一的日志初始化函数

    该函数配置日志系统，同时输出到控制台和文件。

    Args:
        log_name: 日志名称前缀（如 "batch_rerank"）
        log_dir: 日志目录（如为 None，使用当前目录）
        level: 日志级别（默认 logging.INFO）
        format_string: 自定义日志格式（如为 None，使用默认格式）

    Returns:
        配置好的 logger 对象

    Examples:
        >>> logger = setup_logging("my_app", "logs", logging.DEBUG)
        >>> logger.info("Application started")

        >>> logger = setup_logging("rerank")  # 不指定目录，使用当前目录
    """
    # 创建日志文件路径
    if log_dir:
        log_dir_path = ensure_dir(log_dir)
        log_file = log_dir_path / f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_file = Path(f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 日志格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 移除已存在的处理器（避免重复）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"✓ 日志已初始化，日志文件: {log_file}", flush=True)
    except IOError as e:
        print(f"[ERROR] 无法创建日志文件 {log_file}: {e}", flush=True)

    # 返回模块级别的 logger
    logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# 配置管理
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    通用配置文件加载器

    支持 JSON 配置文件的加载，包含完整的错误处理和日志记录。

    Args:
        config_path: 配置文件路径（支持相对路径和绝对路径）

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误

    Examples:
        >>> config = load_config("config.json")
        >>> db_url = config['database']['url']

        >>> config = load_config("/absolute/path/config.json")
    """
    config_file = Path(config_path)
    logger = logging.getLogger(__name__)

    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"✓ 配置已加载: {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {config_path}")
        logger.error(f"  错误详情: {e}")
        raise


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    验证配置是否包含所有必需的键

    Args:
        config: 配置字典
        required_keys: 必需键列表

    Returns:
        True 如果所有必需键都存在

    Raises:
        ValueError: 如果缺少必需键
    """
    logger = logging.getLogger(__name__)
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        logger.error(f"配置缺少必需键: {missing_keys}")
        raise ValueError(f"Missing required config keys: {missing_keys}")

    return True


# ============================================================================
# JSON 文件操作
# ============================================================================

def save_json(
    data: Any,
    output_path: str,
    ensure_ascii: bool = False,
    indent: int = 2,
    create_dir: bool = True
) -> Path:
    """
    统一的 JSON 保存函数

    支持自动目录创建和详细的日志记录。

    Args:
        data: 要保存的数据
        output_path: 输出文件路径
        ensure_ascii: 是否保证 ASCII 编码（默认 False，支持 UTF-8）
        indent: JSON 缩进级别（默认 2）
        create_dir: 是否自动创建父目录（默认 True）

    Returns:
        保存的文件路径 (Path 对象)

    Raises:
        IOError: 文件保存失败

    Examples:
        >>> results = [{'id': 1, 'name': 'test'}]
        >>> save_json(results, "output.json")

        >>> config = {'debug': True}
        >>> save_json(config, "config.json", indent=4)
    """
    file_path = Path(output_path)
    logger = logging.getLogger(__name__)

    if create_dir:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)

        # 计算数据量信息
        data_info = ""
        if isinstance(data, list):
            data_info = f" ({len(data)} 项)"
        elif isinstance(data, dict):
            data_info = f" ({len(data)} 键)"

        logger.info(f"✓ JSON 已保存: {file_path}{data_info}")
        return file_path
    except IOError as e:
        logger.error(f"✗ JSON 保存失败 {output_path}: {e}")
        raise


def load_json(file_path: str) -> Any:
    """
    统一的 JSON 加载函数

    支持完整的错误处理和日志记录。

    Args:
        file_path: JSON 文件路径

    Returns:
        解析后的数据

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 格式错误

    Examples:
        >>> data = load_json("data.json")
        >>> print(f"加载了 {len(data)} 条记录")

        >>> config = load_json("config.json")
    """
    path = Path(file_path)
    logger = logging.getLogger(__name__)

    if not path.exists():
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 计算数据量信息
        data_info = ""
        if isinstance(data, list):
            data_info = f" ({len(data)} 项)"
        elif isinstance(data, dict):
            data_info = f" ({len(data)} 键)"

        logger.info(f"✓ JSON 已加载: {file_path}{data_info}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON 格式错误 {file_path}: {e}")
        raise


# ============================================================================
# 文件和路径操作
# ============================================================================

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径（字符串或 Path 对象）

    Returns:
        Path 对象

    Examples:
        >>> output_dir = ensure_dir("./results")
        >>> ensure_dir("/absolute/path/to/dir")
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_timestamped_filename(name: str, extension: str = "json") -> str:
    """
    生成带时间戳的文件名

    格式: {name}_{YYYYMMDD}_{HHMMSS}.{extension}

    Args:
        name: 文件名前缀
        extension: 文件扩展名（不包含点号）

    Returns:
        时间戳文件名

    Examples:
        >>> filename = get_timestamped_filename("results")
        >>> print(filename)  # 输出: results_20260320_145300.json

        >>> log_file = get_timestamped_filename("app", "log")
        >>> print(log_file)  # 输出: app_20260320_145300.log
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{name}_{timestamp}.{extension}"


def get_backup_filename(original_path: str) -> str:
    """
    为文件生成备份文件名

    格式: {name}.backup_{YYYYMMDD}_{HHMMSS}.{extension}

    Args:
        original_path: 原始文件路径

    Returns:
        备份文件名

    Examples:
        >>> backup_name = get_backup_filename("data.json")
        >>> print(backup_name)  # 输出: data.backup_20260320_145300.json
    """
    path = Path(original_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{path.stem}.backup_{timestamp}{path.suffix}"


# ============================================================================
# 进度追踪和统计
# ============================================================================

@dataclass
class ProcessingStats:
    """
    处理统计信息

    用于追踪批处理过程中的进度、成功率等指标。

    Attributes:
        total: 总处理项数
        successful: 成功处理项数
        failed: 失败处理项数
        skipped: 跳过的项数
        start_time: 处理开始时间（秒时间戳）
    """
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()

    def get_elapsed_seconds(self) -> float:
        """获取已用时间（秒）"""
        return time.time() - self.start_time

    def get_elapsed_minutes(self) -> float:
        """获取已用时间（分钟）"""
        return self.get_elapsed_seconds() / 60

    def get_eta_seconds(self, current: int) -> float:
        """
        获取剩余时间（秒）

        基于当前处理速度估算剩余时间。

        Args:
            current: 当前已处理项数

        Returns:
            预计剩余时间（秒）
        """
        if current == 0:
            return 0
        elapsed = self.get_elapsed_seconds()
        rate = elapsed / current
        return rate * (self.total - current)

    def get_eta_minutes(self, current: int) -> float:
        """获取剩余时间（分钟）"""
        return self.get_eta_seconds(current) / 60

    def get_success_rate(self) -> float:
        """获取成功率（百分比）"""
        processed = self.successful + self.failed
        if processed == 0:
            return 0
        return (self.successful / processed) * 100

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            **asdict(self),
            "elapsed_seconds": self.get_elapsed_seconds(),
            "elapsed_minutes": self.get_elapsed_minutes(),
            "success_rate": self.get_success_rate()
        }

    def __str__(self) -> str:
        """字符串表示"""
        elapsed = self.get_elapsed_minutes()
        return (f"Stats(total={self.total}, successful={self.successful}, "
                f"failed={self.failed}, skipped={self.skipped}, "
                f"elapsed={elapsed:.1f}min, rate={self.get_success_rate():.1f}%)")


@dataclass
class FailedItem:
    """
    失败项记录

    记录处理失败的项目及其错误信息。

    Attributes:
        item_id: 项目 ID
        error: 错误消息
        error_type: 错误类型（异常类名）
        timestamp: 失败时间戳
    """
    item_id: str
    error: str
    error_type: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "item_id": self.item_id,
            "error": self.error,
            "error_type": self.error_type,
            "timestamp": self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """字符串表示"""
        return f"FailedItem(id={self.item_id}, error={self.error[:50]}...)"


# ============================================================================
# 时间和格式化工具
# ============================================================================

def format_duration(seconds: float) -> str:
    """
    格式化时间间隔

    Args:
        seconds: 时间间隔（秒）

    Returns:
        格式化的时间字符串

    Examples:
        >>> format_duration(3661)
        '1h 1m 1s'

        >>> format_duration(125)
        '2m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def log_with_timestamp(message: str, level: str = "INFO") -> None:
    """
    输出带时间戳的日志消息

    这是一个简单的日志函数，不依赖 logging 模块配置。
    适合在早期初始化阶段使用。

    Args:
        message: 日志消息
        level: 日志级别（默认 "INFO"）

    Examples:
        >>> log_with_timestamp("Processing started")
        [2026-03-20 14:53:00] INFO - Processing started

        >>> log_with_timestamp("Error occurred", "ERROR")
        [2026-03-20 14:53:01] ERROR - Error occurred
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level} - {message}", flush=True)


# ============================================================================
# 批处理管理器
# ============================================================================

class BatchProcessor:
    """
    通用批处理管理器

    提供统一的批处理框架，包括：
    - 进度追踪
    - 失败项管理
    - 自动保存中间结果
    - 详细的日志记录

    Examples:
        >>> processor = BatchProcessor(total=100, save_interval=25)
        >>> for item in items:
        ...     try:
        ...         result = process_item(item)
        ...         processor.record_success(item['id'])
        ...     except Exception as e:
        ...         processor.record_failure(item['id'], str(e))
        ...     finally:
        ...         processor.maybe_save()
    """

    def __init__(
        self,
        total: int,
        output_dir: Union[str, Path] = "results",
        save_interval: int = 50,
        log_interval: int = 10
    ):
        """
        初始化批处理器

        Args:
            total: 总处理项数
            output_dir: 输出目录（自动创建）
            save_interval: 每 N 次成功后保存中间结果（0 表示禁用）
            log_interval: 每 N 次后输出一次进度日志
        """
        self.logger = logging.getLogger(__name__)
        self.stats = ProcessingStats(total=total)
        self.failed_items: List[FailedItem] = []
        self.output_dir = ensure_dir(output_dir)
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.processed_count = 0
        self.last_save_count = 0

    def record_success(self, item_id: str) -> None:
        """记录成功处理的项目"""
        self.stats.successful += 1
        self.processed_count += 1

    def record_failure(self, item_id: str, error: str, error_type: Optional[str] = None) -> None:
        """记录失败的项目"""
        self.stats.failed += 1
        self.processed_count += 1
        self.failed_items.append(FailedItem(item_id, error, error_type))

    def record_skipped(self) -> None:
        """记录跳过的项目"""
        self.stats.skipped += 1
        self.processed_count += 1

    def maybe_save(self, force: bool = False) -> None:
        """
        如果达到保存间隔则保存中间结果

        Args:
            force: 强制保存（即使未达到保存间隔）
        """
        if force or (
            self.save_interval > 0
            and self.stats.successful - self.last_save_count >= self.save_interval
        ):
            self._save_intermediate()
            self.last_save_count = self.stats.successful

    def maybe_log_progress(self, force: bool = False) -> None:
        """
        如果达到日志间隔则输出进度

        Args:
            force: 强制输出（即使未达到日志间隔）
        """
        if force or (self.processed_count % self.log_interval == 0):
            progress = (self.processed_count / self.stats.total) * 100
            eta_min = self.stats.get_eta_minutes(self.processed_count)
            self.logger.info(
                f"进度: {self.processed_count}/{self.stats.total} ({progress:.1f}%) | "
                f"成功: {self.stats.successful} | 失败: {self.stats.failed} | "
                f"预计剩余: {eta_min:.1f} 分钟"
            )

    def _save_intermediate(self) -> None:
        """保存中间结果"""
        # 保存统计信息
        stats_file = self.output_dir / f"stats_{self.processed_count}.json"
        save_json(self.stats.to_dict(), str(stats_file))

        # 保存失败项
        if self.failed_items:
            failures_file = self.output_dir / f"failures_{self.processed_count}.json"
            save_json(
                [item.to_dict() for item in self.failed_items],
                str(failures_file)
            )

    def finalize(self) -> Dict[str, Any]:
        """
        完成批处理并保存最终结果

        Returns:
            最终统计信息
        """
        # 最后的保存和日志
        self.maybe_save(force=True)
        self.maybe_log_progress(force=True)

        # 保存最终统计
        final_stats = self.stats.to_dict()
        final_stats['failed_items'] = [item.to_dict() for item in self.failed_items]

        stats_file = self.output_dir / "final_stats.json"
        save_json(final_stats, str(stats_file))

        self.logger.info("=" * 80)
        self.logger.info(f"批处理完成: {self.stats}")
        self.logger.info("=" * 80)

        return final_stats


if __name__ == "__main__":
    # 测试代码
    logger = setup_logging("test", "logs")
    logger.info("Common utilities module loaded successfully")

    # 测试配置加载
    try:
        # 这会因为文件不存在而失败，但演示了用法
        config = load_config("test_config.json")
    except FileNotFoundError:
        logger.info("Test config file not found (expected)")

    # 测试 JSON 保存/加载
    test_data = {"test": "data", "numbers": [1, 2, 3]}
    test_file = ensure_dir("test_output") / "test.json"
    save_json(test_data, str(test_file))
    loaded_data = load_json(str(test_file))
    logger.info(f"Saved and loaded test data: {loaded_data}")

    # 测试时间戳文件名
    filename = get_timestamped_filename("test_results")
    logger.info(f"Timestamped filename: {filename}")

    # 测试处理统计
    stats = ProcessingStats(total=100)
    stats.successful = 85
    stats.failed = 15
    logger.info(f"Processing stats: {stats}")

    logger.info("All tests completed successfully")
