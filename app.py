import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from rag_engine import retrieve, generate_marketing_copy, client
from agent_engine import evaluate_copy, rewrite_copy

st.set_page_config(page_title="营销内容优化 Agent", layout="centered")
st.title("🎯 营销内容优化 Agent")
st.markdown("基于 RAG + Agentic Loop 的营销文案生成与评估系统")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 参数设置")
    brand = st.selectbox("选择品牌", ["olay"], index=0)
    max_iter = st.slider("最大重写次数", 1, 5, 3)
    pass_score = st.slider("合格分数线", 50, 100, 70)
    st.markdown("---")
    st.markdown("**技术栈**：DeepSeek API + BGE-small-zh + FAISS + Streamlit")

# 主界面输入
user_query = st.text_area(
    "请输入你的营销需求：", 
    value="帮我写一条小红书文案，推广OLAY小白瓶精华，针对熬夜暗沉的上班族，强调提亮效果和性价比。",
    height=100
)

if st.button("🚀 开始生成", type="primary"):
    if not user_query.strip():
        st.warning("请输入需求！")
    else:
        # 1. 检索
        with st.status("🔍 正在检索知识库...", expanded=True) as status:
            docs = retrieve(user_query, brand_name=brand)
            context = "\n".join([f"[{doc[0]}] {doc[1]}" for doc in docs])
            for doc in docs:
                st.write(f"命中来源：{doc[0]}")
            status.update(label="✅ 知识库检索完成", state="complete", expanded=False)
        
        # 2. 生成初稿
        with st.spinner("✍️ 正在生成文案初稿..."):
            current_copy = generate_marketing_copy(user_query, docs)
        
        iteration = 0
        best_score = 0
        best_copy = current_copy
        best_eval = None
        history = []
        
        # 3. 多轮评估循环
        while iteration < max_iter:
            iteration += 1
            with st.spinner(f"🧐 正在评估第 {iteration} 版文案..."):
                eval_result = evaluate_copy(user_query, current_copy)
                score = eval_result["total"]
                
                if score >= best_score:
                    best_score = score
                    best_copy = current_copy
                    best_eval = eval_result
                
                history.append({"iteration": iteration, "score": score, "eval": eval_result})
                
            if score >= pass_score:
                st.success(f"第 {iteration} 版评分合格（{score}分），结束迭代！")
                break
            else:
                if iteration < max_iter:
                    st.warning(f"第 {iteration} 版评分较低（{score}分），启动重写...")
                    feedback = eval_result["raw"].split("改进建议：")[-1] if "改进建议：" in eval_result["raw"] else eval_result["raw"]
                    current_copy = rewrite_copy(user_query, context, current_copy, feedback)
                else:
                    st.error(f"已达最大迭代次数，输出历史最高分版本。")

        # 4. 输出最终结果
        st.markdown("---")
        st.subheader(f"🏆 最终生成结果（历史最高评分：{best_score} 分）")
        st.markdown(best_copy)
        
        # 5. 分项得分展示
        if best_eval:
            st.markdown("### 📊 分项得分")
            col1, col2, col3 = st.columns(3)
            col1.metric("品牌合规性", f"{best_eval['compliance']}/40")
            col2.metric("平台适配度", f"{best_eval['platform']}/30")
            col3.metric("吸引力", f"{best_eval['attraction']}/30")
        
        # 6. 历次评估报告
        with st.expander("📊 查看历次评估报告与迭代过程"):
            for h in history:
                st.markdown(f"**第 {h['iteration']} 版（评分：{h['score']}）**")
                st.text(h["eval"]["raw"])
                st.markdown("---")
        
        st.balloons()