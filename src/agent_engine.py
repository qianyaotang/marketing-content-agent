import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
from rag_engine import retrieve, generate_marketing_copy, client

# ========== 1. 结构化评估工具 ==========
def evaluate_copy(user_query, copywriting):
    prompt = f"""你是一名严格的营销总监。请对以下文案进行逐项打分。

【用户需求】
{user_query}

【待评估文案】
{copywriting}

【评分表】
请严格按照以下评分表逐项打分，每项扣分必须给出具体依据：

一、品牌合规性（满分40分）
1. 无绝对化用语（15分）：出现"最""第一""绝对""100%"等，直接扣15分
2. 无医疗功效承诺（10分）：出现"根治""治愈""药到病除"等，直接扣10分
3. 无虚假/夸大宣传（10分）：出现"永久""彻底""秒吸收""不闷痘"等软性夸大，酌情扣3-10分
4. 品牌调性一致（5分）：语气不符品牌指南，扣3-5分

二、平台适配度（满分30分）
5. 标题/开头钩子（10分）：前3秒无痛点钩子，扣5-10分
6. 内容结构完整（10分）：缺使用场景/价格锚点/互动引导各扣3分
7. 互动引导设计（5分）：无评论区引导，扣5分
8. 标签/关键词（5分）：标签少于3个或无关，扣3-5分

三、吸引力（满分30分）
9. 痛点精准度（10分）：痛点模糊，扣5-10分
10. 情绪张力/记忆点（10分）：平铺直叙无爆点，扣5-10分
11. 差异化/独特卖点（10分）：同质化严重，扣5-10分

【输出格式】
请严格按照以下格式输出，不要添加额外解释：

合规性得分：XX/40
平台适配得分：XX/30
吸引力得分：XX/30
综合评分：XX
理由：（逐项说明扣分依据，每条一行）
改进建议：（针对扣分项给出具体修改建议，每条一行）
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位严苛但公正的营销总监，严格按照评分表逐项打分，不凭感觉给分。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        timeout=60
    )
    raw_result = response.choices[0].message.content

    # 结构化解析
    def extract_score(pattern, text):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    compliance = extract_score(r'合规性得分[：:]\s*(\d+)', raw_result)
    platform = extract_score(r'平台适配得分[：:]\s*(\d+)', raw_result)
    attraction = extract_score(r'吸引力得分[：:]\s*(\d+)', raw_result)
    total = extract_score(r'综合评分[：:]\s*(\d+)', raw_result)

    # 兜底：如果综合评分没提取到，用分项加总
    if total == 0:
        total = compliance + platform + attraction

    return {
        "raw": raw_result,
        "compliance": compliance,
        "platform": platform,
        "attraction": attraction,
        "total": total
    }

# ========== 2. 重写工具 ==========
def rewrite_copy(user_query, context, previous_copy, feedback):
    prompt = f"""你是一名专业的营销文案专家。请根据修改意见，重写以下营销文案。

【用户需求】
{user_query}

【品牌与产品信息】
{context}

【之前生成的文案】
{previous_copy}

【修改意见】
{feedback}

【核心要求】
1. 必须严格遵守修改意见，严禁在修改时引入任何新的绝对化用语、医疗暗示、夸大功效词汇。
2. 严格控制"秒吸收""不闷痘""救星""回血神器"等宣称，如需表达请改为"肤感清爽，不易给油皮增加负担"等合规表述。
3. 保持原有文案的网感，但所有表述必须合规、克制。
4. 直接输出重写后的文案正文，不需要解释过程。
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是专业文案，非常善于根据反馈修改文案，且严格遵守合规底线。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
	 timeout=60
    )
    return response.choices[0].message.content

# ========== 3. Agent 核心循环 ==========
def run_agent(user_query, brand_name="olay", max_iterations=3, pass_score=70):
    print(f"\n品牌: {brand_name}")
    print(f"用户需求: {user_query}\n")
    
    # Step 1: 检索指定品牌的知识库
    print(">>> [Agent 思考] 正在检索知识库...")
    docs = retrieve(user_query, brand_name=brand_name)
    context = "\n".join([f"[{doc[0]}] {doc[1]}" for doc in docs])
    for doc in docs:
        print(f"   - 命中来源: {doc[0]}")
        
    # Step 2: 生成初稿
    print("\n>>> [Agent 行动] 正在生成文案初稿...")
    current_copy = generate_marketing_copy(user_query, docs)
    
    iteration = 0
    best_score = 0
    best_copy = current_copy
    best_eval = None
    history = []
    
    # Step 3: 多轮评估循环
    while iteration < max_iterations:
        iteration += 1
        print(f"\n>>> [Agent 思考] 正在评估第 {iteration} 版文案...")
        eval_result = evaluate_copy(user_query, current_copy)
        print(f"--- 第 {iteration} 版评估报告 ---")
        print(eval_result["raw"])
        print("----------------------")
        
        score = eval_result["total"]
        
        # 记录历史最高分版本（同分取最新）
        if score >= best_score:
            best_score = score
            best_copy = current_copy
            best_eval = eval_result
            
        history.append({"iteration": iteration, "score": score, "eval": eval_result})
            
        # 判断是否达标
        if score >= pass_score:
            print(f"\n>>> [Agent 决策] 第 {iteration} 版评分合格（{score}分），结束迭代！")
            break
        else:
            print(f"\n>>> [Agent 决策] 第 {iteration} 版评分较低（{score}分）。")
            if iteration < max_iterations:
                print(f">>> [Agent 行动] 启动重写工具，进行第 {iteration + 1} 次生成...")
                feedback = eval_result["raw"].split("改进建议：")[-1] if "改进建议：" in eval_result["raw"] else eval_result["raw"]
                current_copy = rewrite_copy(user_query, context, current_copy, feedback)
            else:
                print(f">>> [Agent 决策] 已达最大迭代次数（{max_iterations}次），输出历史最佳版本。")
    
    # Step 4: 输出最终结果
    print(f"\n===== 最终生成结果（历史最高评分：{best_score}分） =====")
    print(best_copy)
    print("========================")
    
    return {
        "final_copy": best_copy,
        "best_score": best_score,
        "best_eval": best_eval,
        "history": history
    }

if __name__ == "__main__":
    test_query = "帮我写一条小红书文案，推广OLAY小白瓶精华，针对熬夜暗沉的上班族，强调提亮效果和性价比。"
    run_agent(test_query, brand_name="olay", pass_score=70)