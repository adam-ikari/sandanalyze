"""Shared test fixtures for sandanalyze."""
import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_grain_image():
    """生成一张模拟沙粒图像：深灰背景上多个亮色椭圆颗粒。"""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[:] = 40  # 深灰背景

    # 画3个不同大小的椭圆模拟沙粒
    img = cv2.ellipse(img, (50, 50), (15, 10), 30, 0, 360, 180, -1)
    img = cv2.ellipse(img, (130, 60), (20, 12), -20, 0, 360, 180, -1)
    img = cv2.ellipse(img, (80, 140), (12, 12), 0, 0, 360, 180, -1)
    return img


@pytest.fixture
def overlapping_grain_image():
    """生成一张有粘连颗粒的图像，用于测试分水岭。"""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[:] = 40
    # 两个靠近的椭圆模拟粘连颗粒
    img = cv2.ellipse(img, (60, 100), (20, 12), 0, 0, 360, 180, -1)
    img = cv2.ellipse(img, (100, 100), (20, 12), 0, 0, 360, 180, -1)
    return img


@pytest.fixture
def real_sand_image_path():
    """返回项目中的真实沙粒图像路径。"""
    return "Sand_from_Gobi_Desert.jpg"
