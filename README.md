# SandAnalyze - 沙粒形态分析系统

基于 OpenCV 传统图像处理和 YOLOv8-seg 的沙粒识别、形状分析和数量统计工具。

## 功能

- 🔬 **沙粒检测**：传统 OpenCV 轮廓检测 + YOLOv8-seg 对比
- 📐 **多维度形态参数**：圆度、球度、长短轴比、凸度、Feret 直径等
- 📊 **统计分析**：粒径分布、圆度-球度散点图、Zingg 分类
- 🌐 **Web 界面**：Streamlit 交互界面，浏览器中运行
- 📤 **导出**：CSV 数据 + 标注图 PNG

## 安装

```bash
uv sync --extra dev
```

## 使用

```bash
# 启动 Web 应用
uv run python main.py

# 或直接使用 streamlit
streamlit run app.py

# 指定端口
uv run python main.py --server.port 8080
```

1. 在侧边栏上传沙粒图像
2. 调整预处理参数
3. 点击"运行检测"
4. 查看统计摘要、颗粒数据表格和 Plotly 交互式图表
5. 导出 CSV 或标注图

## 形态参数

| 参数 | 计算方法 | 地质意义 |
|------|----------|----------|
| 面积 (A) | 掩码像素数 | 粒径基础 |
| 周长 (P) | 轮廓长度 | 磨蚀程度 |
| 圆度 (Circularity) | 4πA/P² | 越接近1越圆 |
| 等效粒径 (d_eq) | √(4A/π) | 等效圆直径 |
| 长短轴比 (AR) | 长/短轴 | 扁平程度 |
| 球度 (Sphericity) | 短/长轴 | 三维形状推断 |
| 凸度 (Convexity) | 面积/凸包面积 | 表面凹凸程度 |

## 技术栈

- Python 3.13
- OpenCV（图像处理）
- ultralytics / YOLOv8-seg（深度学习检测）
- Streamlit（Web UI）
- Plotly（交互式图表）
- numpy / scipy（数值计算）
