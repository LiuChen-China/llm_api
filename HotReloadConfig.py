import os
import yaml
import time
from typing import Any, Dict
from threading import Lock  # 新增：线程安全锁

class HotReloadConfig:
    """
    自动热加载 YAML 配置类
    每次访问属性时，自动检查配置文件是否修改，修改则重新加载
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """
        __new__ 是 Python 创建实例的方法
        这里控制：永远只返回同一个实例
        """
        with cls._lock:  # 加锁，防止多线程创建多个实例
            if cls._instance is None:
                # 第一次创建：正常初始化
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str, check_interval: float = 3.0):
        '''
        :param config_path: 配置文件路径
        :param check_interval: 检查间隔（秒），默认 1 秒
        '''
        if "_initialized" in self.__dict__:
            return
        self._CONFIG_PATH = config_path# 配置文件路径
        self._CHECK_INTERVAL = check_interval# 最小检查间隔（秒）
        self._config_data: Dict[str, Any] = {}
        self._last_check_time = 0.0
        # 初始化加载一次
        self._load_config(force=True)
        self._initialized = True#就算是单例模式，只要实例化一次都会执行一次__init__
        print(f'初始化配置文件：{self._CONFIG_PATH}')

    def _load_config(self, force: bool = False) -> None:
        """
        加载/重新加载配置文件
        :param force: 是否强制加载，忽略检查间隔，默认 False
        """
        # 优化：控制检查频率，避免高频 IO
        now = time.time()
        if (not force) and (now - self._last_check_time) < self._CHECK_INTERVAL:
            return
        self._last_check_time = now
        # 文件已修改 → 热加载
        with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
            self._config_data = yaml.safe_load(f) or {}

    def _get_nested_value(self, keys: list) -> Any:
        """获取嵌套配置值"""
        data = self._config_data
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return None
        return data

    def __getattr__(self, name: str) -> Any:
        """
        核心：属性访问时自动触发热加载
        支持链式调用：config.redis.host
        """
        # 每次访问属性前，先检查并加载配置
        self._load_config()
        # 处理链式属性（递归返回子配置对象）
        value = self._get_nested_value([name])
        if isinstance(value, dict):
            return self._SubConfig(self, [name])
        return value

    class _SubConfig:
        """内部类：支持链式访问嵌套配置"""
        def __init__(self, parent: "HotReloadConfig", path: list):
            self._parent = parent
            self._path = path

        def __getattr__(self, name: str) -> Any:
            self._parent._load_config()
            new_path = self._path + [name]
            value = self._parent._get_nested_value(new_path)
            if isinstance(value, dict):
                return HotReloadConfig._SubConfig(self._parent, new_path)
            return value

config = HotReloadConfig("config.yaml")
