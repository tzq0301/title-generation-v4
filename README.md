# 2022 中软杯 A9 智能创作平台 标题生成算法模型

基于 bert4keras 与 T5-pegasus 进行标题生成模型的训练与微调

> T5 Pegasus 是追一科技参考 Pegasus 为 T5 设计的预训练任务，且使用 jieba 作为分词器，对中文预料更加友好

模型经过压缩后，使用 ONNX 进行导出，提升在生产环境的推理速度，降低响应时延

可在根目录下使用 Docker 容器直接运行

## Rouge 指标

- Rouge-1: 0.4055
- Rouge-2: 0.3019
- Rouge-L: 0.3869

## 效果对比

| 原标题 | t5-pegasus | imxly/t5-copy-summary |
| ---- | ---- | ---- |
| 商务部：**全力保障汛期生活必需品市场平稳运行** | **全力保障汛期生活必需品市场平稳运行** | 商务部：将继续密切跟踪汛情对生活必需品市场供应的影响 |
| 退役军人事务部等8部门联合印发意见：**全面启动优抚医院改革** | **全面启动优抚医院改革** | 三部门联合印发《关于推进优抚医院改革发展的意见》推进医院改革 |
| **国企办医疗机构改革任务基本完成** | **国**有**企**业**办医疗机构改革任务基本完成** | 国有企业办医疗机构深化改革任务已基本完成 |

## 部署响应时间

在本地测试环境（不考虑网络）下：

| 操作系统（OS） | CPU 核心数 | CPU 型号 | 平均响应时间（s） | 
| ---- | ---- | ---- | ---- |
| Ubuntu 20	| 8	Intel(R) Xeon(R) | CPU E5-2680 v4 @ 2.40GHz | **0.6220** |
| Windows 11 | 8 | 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz | 1.3174 |
