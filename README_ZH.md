# LLM 与 Ising 结构发现

本项目用可检验的广义 Ising 模拟实验，研究语言模型能否从「提出假设—预测—实验—更新」的教学记录中学会恢复隐藏相互作用。

数值搜索程序充当老师，Qwen 模型充当学生。A 组获得初始观测和最终监督答案；B 组增加完整实验与观测；C 组进一步增加实验前的假设、数值预测和观察后的更新。B/C 使用相同的实验、观测和标签。

**当前发布的是研究代码。正式三种教学条件的学生对照实验尚未完成，不能据此声称过程教学有效。** 开发阶段也存在负结果；输入预处理、接口练习及参数多样性尝试没有证明机制恢复能力得到改善。

## 从零运行

使用 Python 3.12，在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-discovery-numeric-lock.txt
.venv\Scripts\python.exe public_checks.py
.venv\Scripts\python.exe demo_teaching.py --output output/teaching_demo
```

示例只依赖 NumPy，生成一条带噪声的教学轨迹，重放核验 24 轮实验选择和预测，导出三组教材及通用聊天消息。`inference_request.json` 是只含已观察证据的学生提示；完整 A/B/C 消息含教学监督标签，应与推理输入区分。

数值搜索的小规模比较：

```powershell
.venv\Scripts\python.exe run_structure_discovery.py --worlds-per-family 1 --eval-count 32 --output results/search_demo
```

以上两项无需模型、GPU 或 API 密钥。模型训练需要额外的 PyTorch/CUDA 环境；历史训练脚本依赖生成数据、检查点及源文件哈希，详见 [实验入口说明](PROJECT_GUIDE.md)。公开仓库不包含模型权重、原始运行目录或未完成论文。

该任务限定在有限候选作用语法内，使用预先设置的自旋构型。它不等于开放式物理定律发现；数值基线的准确率也不等于语言模型的成绩。
