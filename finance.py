import streamlit as st
import pandas as pd
import datetime
from datetime import date
import requests
import json
import base64
from io import StringIO
import os

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
# 推荐的视觉模型，SiliconFlow 上可用
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
# 文本分析模型 (用户指定)
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

# --- 存储类：处理数据保存 ---
class DataManager:
    """数据管理类，支持 GitHub 远程存储和本地 CSV 存储"""
    def __init__(self, github_token=None, repo=None, filename="ledger.csv"):
        self.github_token = github_token
        # 自动处理完整的 GitHub URL，提取 owner/repo
        if repo and repo.startswith("http"):
            self.repo = repo.rstrip("/").split("github.com/")[-1]
        else:
            self.repo = repo
        self.filename = filename
        self.use_github = bool(github_token and self.repo)

    def load_data(self):
        """加载数据"""
        if self.use_github:
            return self._load_from_github()
        else:
            return self._load_from_local()

    def save_data(self, df, sha=None):
        """保存数据"""
        if self.use_github:
            return self._save_to_github(df, sha)
        else:
            return self._save_to_local(df)

    # --- 本地存储逻辑 ---
    def _load_from_local(self):
        if os.path.exists(self.filename):
            return pd.read_csv(self.filename), None
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None

    def _save_to_local(self, df):
        df.to_csv(self.filename, index=False)
        return True

    # --- GitHub 存储逻辑 ---
    def _load_from_github(self):
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            content = response.json()
            csv_str = base64.b64decode(content['content']).decode('utf-8')
            try:
                return pd.read_csv(StringIO(csv_str)), content['sha']
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), content['sha']
        elif response.status_code == 404:
            return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None
        else:
            st.error(f"GitHub 读取错误: {response.status_code}")
            return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None

    def _save_to_github(self, df, sha):
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        csv_str = df.to_csv(index=False)
        content_bytes = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
        
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        data = {
            "message": f"Update ledger {datetime.datetime.now()}",
            "content": content_bytes
        }
        if sha:
            data["sha"] = sha
            
        response = requests.put(url, headers=headers, data=json.dumps(data))
        return response.status_code in [200, 201]

# --- AI 处理函数 ---
def process_bill_image(image_file, api_key):
    if not api_key:
        return None, "未配置 API Key"

    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 针对 Qwen-VL 优化的 Prompt
    prompt = """
    请识别这张账单图片。提取以下字段并以JSON格式返回：
    1. date (格式YYYY-MM-DD)
    2. amount (数字类型，不要带货币符号)
    3. merchant (商户名或交易说明)
    4. category (从以下选择最接近的: 餐饮, 交通, 购物, 居住, 娱乐, 工资, 其他)
    5. type (支出 或 收入)
    
    直接返回JSON，不需要 ```json 标记。
    """

    payload = {
        "model": VISION_MODEL_NAME, 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512
    }

    try:
        response = requests.post(
            "[https://api.siliconflow.cn/v1/chat/completions](https://api.siliconflow.cn/v1/chat/completions)",
            headers=headers,
            json=payload,
            timeout=45
        )
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # 清洗数据，防止 markdown 干扰
            clean_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_content), None
        else:
            return None, f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"请求异常: {str(e)}"

# --- 主程序 ---
def main():
    # 1. 配置加载
    st.sidebar.title("⚙️ 个人财务设置")
    
    # 获取配置 (优先读取 secrets.toml)
    sf_api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    github_token = st.secrets.get("GITHUB_TOKEN", "")
    github_repo = st.secrets.get("GITHUB_REPO", "")

    # 初始化存储管理器
    dm = DataManager(github_token, github_repo)
    
    # 侧边栏状态指示
    if dm.use_github:
        st.sidebar.success(f"☁️ 数据存储: GitHub ({github_repo})")
    else:
        st.sidebar.warning("📂 数据存储: 本地模式 (重启后Streamlit Cloud会重置数据)")

    # 财务设置
    payday = st.sidebar.number_input("每月发薪日", 1, 31, 10)
    current_cash = st.sidebar.number_input("当前现金/余额", value=3000.0)

    # 2. 加载数据
    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    # 3. 财务概览 (Dashboard)
    st.title("💰 极简账本")
    
    # 计算逻辑
    today = date.today()
    if today.day >= payday:
        next_pay_date = date(today.year + (1 if today.month == 12 else 0), 1 if today.month == 12 else today.month + 1, payday)
    else:
        next_pay_date = date(today.year, today.month, payday)
    
    days_left = (next_pay_date - today).days
    
    # 核心指标
    col1, col2, col3 = st.columns(3)
    col1.metric("当前余额", f"¥{current_cash:,.2f}")
    col2.metric("距离发工资", f"{days_left} 天")
    
    if days_left > 0:
        daily_budget = current_cash / days_left
        gap = daily_budget - DEFAULT_TARGET_SPEND
        col3.metric("每日可用", f"¥{daily_budget:.1f}", 
                    f"{gap:+.1f} (vs ¥{DEFAULT_TARGET_SPEND})",
                    delta_color="normal" if gap >= 0 else "inverse")
        
        if gap < 0:
            st.error(f"⚠️ 警报：每天亏空 {abs(gap):.1f} 元，体面生活岌岌可危！")
        else:
            st.success(f"🎉 状态良好：每天还有 {gap:.1f} 元的“挥霍”空间。")
    else:
        col3.metric("每日可用", "N/A", "今日发薪！")

    st.divider()

    # 4. 记账功能区
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("📸 截图记账 (AI)")
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], key="ocr_upload")
        
        if uploaded_file and st.button("开始识别", key="btn_ocr"):
            if not sf_api_key:
                st.error("请先在 secrets.toml 配置 SILICONFLOW_API_KEY")
            else:
                with st.spinner("AI 正在提取账单信息..."):
                    data, err = process_bill_image(uploaded_file, sf_api_key)
                    if err:
                        st.error(err)
                    else:
                        st.success("识别成功，请在右侧确认添加")
                        st.session_state.temp_ocr_data = data

    with c2:
        st.subheader("📝 确认/手动记账")
        
        # 预填充数据
        default_date = date.today()
        default_amt = 0.0
        default_mer = ""
        default_cat = "餐饮"
        default_type_idx = 0

        if 'temp_ocr_data' in st.session_state:
            res = st.session_state.temp_ocr_data
            try:
                default_date = pd.to_datetime(res.get('date', str(date.today()))).date()
                default_amt = float(res.get('amount', 0.0))
                default_mer = res.get('merchant', '')
                default_cat = res.get('category', '其他')
                default_type_idx = 1 if res.get('type') == '收入' else 0
            except:
                pass

        with st.form("entry_form"):
            f_date = st.date_input("日期", default_date)
            cols = st.columns(2)
            f_type = cols[0].selectbox("类型", ["支出", "收入"], index=default_type_idx)
            f_cat = cols[1].text_input("分类", default_cat)
            f_amt = st.number_input("金额", value=default_amt, step=0.1)
            f_desc = st.text_input("备注/商户", default_mer)
            
            if st.form_submit_button("💾 保存记录"):
                new_row = {
                    "日期": str(f_date), 
                    "类型": f_type, 
                    "金额": f_amt, 
                    "备注": f_desc, 
                    "分类": f_cat
                }
                st.session_state.ledger_data = pd.concat(
                    [st.session_state.ledger_data, pd.DataFrame([new_row])], 
                    ignore_index=True
                )
                
                # 自动保存
                if dm.save_data(st.session_state.ledger_data, st.session_state.get('github_sha')):
                    st.success("已保存！")
                    # 重新加载以获取最新 sha (如果用 GitHub)
                    if dm.use_github:
                        _, new_sha = dm.load_data()
                        st.session_state.github_sha = new_sha
                    if 'temp_ocr_data' in st.session_state:
                        del st.session_state.temp_ocr_data
                    st.rerun()
                else:
                    st.error("保存失败，请检查配置")

    st.divider()

    # 5. 历史账单 (可编辑)
    st.subheader("📊 历史账单 (可直接修改)")
    
    if not st.session_state.ledger_data.empty:
        # 使用 data_editor 允许直接修改表格
        edited_df = st.data_editor(
            st.session_state.ledger_data,
            num_rows="dynamic", # 允许添加/删除行
            use_container_width=True,
            key="history_editor"
        )

        # 检查是否有修改
        # 简单对比：如果 dataframe 不一样了，显示保存按钮
        # 这里的逻辑是：用户修改完 data_editor，Streamlit 会自动更新 session_state 中的 editor key
        # 我们需要一个显式的按钮来触发“写入磁盘/GitHub”的操作
        
        col_save, col_info = st.columns([1, 4])
        with col_save:
            if st.button("🔄 同步修改到存储"):
                if dm.save_data(edited_df, st.session_state.get('github_sha')):
                    st.session_state.ledger_data = edited_df
                    st.success("所有修改已同步！")
                    if dm.use_github:
                         _, new_sha = dm.load_data()
                         st.session_state.github_sha = new_sha
                    st.rerun()
                else:
                    st.error("同步失败")
    else:
        st.info("暂无数据，快去记一笔吧！")

    # 5.5 可视化看板
    if not st.session_state.ledger_data.empty:
        st.divider()
        st.subheader("📈 消费透视")
        
        # 数据预处理
        chart_df = st.session_state.ledger_data.copy()
        # 确保金额是数字，日期是时间格式
        chart_df['金额'] = pd.to_numeric(chart_df['金额'], errors='coerce').fillna(0)
        chart_df['日期'] = pd.to_datetime(chart_df['日期']).dt.date
        
        # 只分析支出数据
        expense_df = chart_df[chart_df['类型'] == '支出']
        
        if not expense_df.empty:
            tab_chart1, tab_chart2 = st.tabs(["📊 分类占比", "📉 每日趋势"])
            
            with tab_chart1:
                # 按分类汇总
                category_sum = expense_df.groupby('分类')['金额'].sum().sort_values(ascending=False)
                st.bar_chart(category_sum, color="#FF4B4B") # 使用红色系代表支出
                
            with tab_chart2:
                # 按日期汇总
                daily_sum = expense_df.groupby('日期')['金额'].sum()
                st.line_chart(daily_sum)
        else:
            st.info("暂无支出数据，记录几笔支出后即可查看图表。")

    # 6. 简单的 AI 分析 (保留)
    with st.expander("🤖 呼叫 AI 财务分析"):
        if st.button("分析我的开销"):
            if sf_api_key and not st.session_state.ledger_data.empty:
                with st.spinner("AI 正在思考..."):
                    summary = st.session_state.ledger_data.to_string()
                    payload = {
                        "model": TEXT_MODEL_NAME, 
                        "messages": [{"role": "user", "content": f"分析这份账单，指出问题：\n{summary}"}]
                    }
                    try:
                        r = requests.post("https://api.siliconflow.cn/v1/chat/completions", 
                                        headers={"Authorization": f"Bearer {sf_api_key}"}, json=payload)
                        st.markdown(r.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"AI 服务异常: {e}")

if __name__ == "__main__":
    main()
