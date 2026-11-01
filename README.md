# 营销内容优化 Agent (Marketing Content Optimization Agent)

基于 RAG (检索增强生成) 与 Agentic Loop (智能体循环) 的企业级营销内容生成与合规评估系统。

## 项目背景

针对品牌营销中“合规风险高、多平台适配难、生成质量不可控”三大痛点，本项目设计并实现了一个多品牌自动化内容生成中台。

## 核心特性

- **多品牌隔离 RAG 架构**：采用 `BGE-small-zh` + `FAISS` 实现毫秒级语义检索，设计“品牌路由层”确保知识不串库，从底层杜绝 AI 幻觉。
- **Agentic Loop 自纠错机制**：设计“生成-评估-重写”多轮闭环。引入结构化评分表（Rubric，3维度11子项），解决大模型评估的“72分安全区死锁”问题。
- **合规防御与工程容错**：建立全局广告法禁用词库，实现“合规一票否决制”；针对 API 响应不稳定，设计超时降级策略，保障系统鲁棒性。
- **业务动态权衡**：UI端暴露“合格分数线”与“最大重写次数”参数，实现 API 成本与内容质量的最佳平衡。

## 技术栈

- **大模型**：DeepSeek API
- **Embedding**：Sentence-Transformers (BGE-small-zh-v1.5)
- **向量库**：FAISS
- **前端框架**：Streamlit
- **编程语言**：Python 3.11

## 本地运行指南

1. 克隆项目并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   在根目录创建 `.env` 文件，填入你的 DeepSeek API Key：
   ```env
   DEEPSEEK_API_KEY=your_api_key_here
   ```

3. 构建向量索引：
   确保 `data` 文件夹下有品牌数据，然后运行：
   ```bash
   python src/build_index.py
   ```

4. 启动 Streamlit 网页应用：
   ```bash
   python -m streamlit run app.py
   ```