import streamlit as st
import pandas as pd
import datetime
from datetime import date
import requests
import json
import base64
from io import StringIO, BytesIO
import os
import pdfplumber
import re

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

# --- 存储类 ---
class DataManager:
    """数据管理类，支持 GitHub 远程存储和本地 CSV 存储"""
    def __init__(self, github_token=None, repo=None, filename="ledger.csv"):
        self.github_token = github_token
        if repo and repo.startswith("http"):
            self.repo = repo.rstrip("/").split("github.com/")[-1]
        else:
            self.repo = repo
        self.filename = filename
        self.use_github = bool(github_token and self.repo)

    def load_data(self):
        if self.use_github:
            return self._load_from_github()
        else:
            return self._load_from_local()

    def save_data(self, df, sha=None):
        if self.use_github:
            return self._save_to_github(df, sha)
        else:
            return self._save_to_local(df)

    def _load_from_local(self):
        if os.path.exists(self.filename):
            try:
                return pd.read_csv(self.filename), None
            except:
                return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"]), None

    def _save_to_local(self, df):
        df.to_csv(self.filename, index=False)
        return True

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

# --- 智能账单解析类 (AI核心版) ---
class BillParser:
    @staticmethod
    def identify_and_parse(file, api_key):
        """智能识别文件类型并提取文本，交给AI解析"""
        if not api_key:
            return None, "请先配置 SILICONFLOW_API_KEY 以使用 AI 解析功能"

        filename = file.name.lower()
        content_text = ""
        source_type = "未知文件"

        try:
            # 1. 提取文件内容为纯文本
            if filename.endswith('.csv'):
                source_type = "CSV账单"
                try:
                    content_text = file.getvalue().decode('utf-8')
                except UnicodeDecodeError:
                    file.seek(0)
                    content_text = file.getvalue().decode('gbk', errors='ignore')
            
            elif filename.endswith(('.xls', '.xlsx')):
                source_type = "Excel账单"
                # 读取Excel所有sheet，转换为CSV字符串拼接
                try:
                    xls = pd.read_excel(file, sheet_name=None)
                    text_parts = []
                    for sheet_name, df in xls.items():
                        # 将DataFrame转为CSV文本，保留上下文结构
                        text_parts.append(f"--- Sheet: {sheet_name} ---\n")
                        text_parts.append(df.to_csv(index=False))
                    content_text = "\n".join(text_parts)
                except Exception as e:
                    return None, f"Excel 读取失败: {e}"

            elif filename.endswith('.pdf'):
                source_type = "PDF账单"
                try:
                    text_parts = []
                    with pdfplumber.open(file) as pdf:
                        for page in pdf.pages:
                            # 优先尝试提取表格
                            tables = page.extract_tables()
                            if tables:
                                for table in tables:
                                    # 将表格转为 CSV 格式文本
                                    df_table = pd.DataFrame(table)
                                    # 清理None
                                    df_table = df_table.fillna("")
                                    text_parts.append(df_table.to_csv(index=False, header=False))
                            else:
                                # 提取纯文本作为兜底
                                text_parts.append(page.extract_text() or "")
                    content_text = "\n".join(text_parts)
                except Exception as e:
                    return None, f"PDF 读取失败: {e}"
            else:
                return None, "不支持的文件格式"

            # 2. 调用 AI 进行解析
            if not content_text.strip():
                return None, "文件内容为空或无法提取文本"
                
            return BillParser._call_ai_parser(content_text, source_type, api_key)

        except Exception as e:
            return None, f"解析过程发生未知错误: {str(e)}"

    @staticmethod
    def _call_ai_parser(content_text, source_type, api_key):
        """调用 DeepSeek-V3.2 进行结构化提取"""
        
        # 截断保护：虽然 DeepSeek 上下文很长，但防止极端大文件，保留前 50000 字符通常足够包含一个月账单的关键信息
        # 如果是CSV，通常头部是关键。如果是流水，最好能处理更多。
        # 这里设置为 100k 字符，DeepSeek处理得过来。
        truncated_content = content_text[:100000]
        
        system_prompt = """
        你是一个专业的财务数据提取助手。你的任务是从杂乱的账单文本中提取交易流水。
        请遵循以下规则：
        1. 输出必须是标准的 JSON 数组格式 `[{"date": "...", ...}, ...]`。
        2. 不要包含 markdown 标记（如 ```json）。
        3. 字段说明：
           - date: 交易日期，格式必须统一为 YYYY-MM-DD。如果年份缺失，默认2025年。
           - type: "支出" 或 "收入"。根据金额正负或"收/支"列判断。通常银行账单中负数是支出，或者在"支出"列的数字。
           - amount: 金额绝对值（数字类型，不要字符串）。
           - merchant: 交易对象/商户名/摘要。
           - category: 根据商户名推断分类（如：餐饮、交通、购物、转账、工资、理财、还款、其他）。
        4. 过滤掉无效行（如表头、页码、统计汇总行、余额行）。只保留具体交易。
        5. 对于"不计收支"或"资金转移"的条目，如果看起来像信用卡还款，标记为"转账"或"还款"，类型自定（通常不记入日常收支，但用户可能需要）。
        6. 如果文本是乱码或无法识别为账单，返回空数组 []。
        """

        user_prompt = f"""
        请处理这份 {source_type} 数据，提取所有交易记录。
        
        数据内容片段：
        {truncated_content}
        """

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": TEXT_MODEL_NAME, # 使用 deepseek-ai/DeepSeek-V3.2
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 8192, # 尽可能输出完整
            "temperature": 0.1  # 低温度保证准确性
        }

        try:
            # 使用 SiliconFlow 兼容接口
            response = requests.post(
                "[https://api.siliconflow.cn/v1/chat/completions](https://api.siliconflow.cn/v1/chat/completions)",
                headers=headers,
                json=payload,
                timeout=120 # 解析大文件需要更多时间
            )
            
            if response.status_code == 200:
                res_json = response.json()
                ai_content = res_json['choices'][0]['message']['content']
                
                # 清洗 Markdown
                ai_content = ai_content.replace("```json", "").replace("```", "").strip()
                
                try:
                    data_list = json.loads(ai_content)
                    if not isinstance(data_list, list):
                        return None, "AI 返回格式错误（非数组）"
                    
                    if not data_list:
                        return None, "AI 未能提取到任何有效交易记录"

                    # 转为 DataFrame 并做基础清洗
                    df = pd.DataFrame(data_list)
                    
                    # 确保列存在
                    required_cols = ["date", "type", "amount", "merchant", "category"]
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = ""
                    
                    # 映射回 app 统一的列名
                    df = df.rename(columns={
                        "date": "日期",
                        "type": "类型",
                        "amount": "金额",
                        "merchant": "备注",
                        "category": "分类"
                    })
                    
                    # 数据类型转换
                    df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
                    # 强制保留 AI 识别出的分类
                    df['分类'] = df['分类'].fillna("AI导入")
                    
                    return df, None
                    
                except json.JSONDecodeError:
                    return None, f"AI 返回了非 JSON 数据: {ai_content[:100]}..."
            else:
                return None, f"API 请求失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            return None, f"AI 请求异常: {str(e)}"

    @staticmethod
    def merge_and_deduplicate(old_df, new_df):
        """
        合并并去重
        """
        if new_df is None or new_df.empty:
            return old_df, 0, 0

        added_rows = []
        skipped_count = 0
        
        existing_keys = set()
        for _, row in old_df.iterrows():
            try:
                amt = float(row['金额'])
                key = f"{row['日期']}_{amt:.2f}_{row['类型']}"
                existing_keys.add(key)
            except:
                continue

        for _, row in new_df.iterrows():
            try:
                amt = float(row['金额'])
                key = f"{row['日期']}_{amt:.2f}_{row['类型']}"
            except:
                continue
            
            # 简单去重逻辑：只要日期、金额、类型完全一致，就认为是重复
            # AI 解析后，备注可能和原始 CSV 不一样，所以不作为去重主键，只作为辅助
            if key in existing_keys:
                skipped_count += 1
                continue
            
            added_rows.append(row)
            existing_keys.add(key) 

        if not added_rows:
            return old_df, 0, skipped_count
            
        return pd.concat([old_df, pd.DataFrame(added_rows)], ignore_index=True), len(added_rows), skipped_count

# --- AI 处理函数 (图片 OCR) ---
def process_bill_image(image_file, api_key):
    if not api_key:
        return None, "未配置 API Key"

    image_bytes = image_file.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

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
        "max_tokens": 1024
    }

    try:
        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45
        )
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
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
    
    sf_api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    github_token = st.secrets.get("GITHUB_TOKEN", "")
    github_repo = st.secrets.get("GITHUB_REPO", "")

    dm = DataManager(github_token, github_repo)
    
    if dm.use_github:
        st.sidebar.success(f"☁️ 数据存储: GitHub ({github_repo})")
    else:
        st.sidebar.warning("📂 数据存储: 本地模式")

    payday = st.sidebar.number_input("每月发薪日", 1, 31, 10)
    current_cash = st.sidebar.number_input("当前现金/余额", value=3000.0)

    # 2. 加载数据
    if 'ledger_data' not in st.session_state:
        df, sha = dm.load_data()
        st.session_state.ledger_data = df
        st.session_state.github_sha = sha

    # 3. 财务概览
    st.title("💰 极简账本")
    
    today = date.today()
    if today.day >= payday:
        next_pay_date = date(today.year + (1 if today.month == 12 else 0), 1 if today.month == 12 else today.month + 1, payday)
    else:
        next_pay_date = date(today.year, today.month, payday)
    
    days_left = (next_pay_date - today).days
    
    col1, col2, col3 = st.columns(3)
    col1.metric("当前余额", f"¥{current_cash:,.2f}")
    col2.metric("距离发工资", f"{days_left} 天")
    
    if days_left > 0:
        daily_budget = current_cash / days_left
        gap = daily_budget - DEFAULT_TARGET_SPEND
        col3.metric("每日可用", f"¥{daily_budget:.1f}", 
                    f"{gap:+.1f} (vs ¥{DEFAULT_TARGET_SPEND})",
                    delta_color="normal" if gap >= 0 else "inverse")
    else:
        col3.metric("每日可用", "N/A", "今日发薪！")

    st.divider()

    # 4. 记账功能区 - 统一入口
    tab_auto, tab_manual = st.tabs(["📤 智能导入 (文件/图片)", "✍️ 手动记账"])

    with tab_auto:
        st.markdown("""
        <small>支持格式：
        1. **图片** (jpg/png) -> 使用 Qwen-VL 视觉模型识别
        2. **文件** (csv/xlsx/xls/pdf) -> 使用 DeepSeek-V3.2 文本模型智能分析 (支持所有银行/支付软件格式)
        </small>
        """, unsafe_allow_html=True)
        
        # 允许上传多个文件
        uploaded_files = st.file_uploader(
            "点击上传 (支持多选)", 
            type=['png', 'jpg', 'jpeg', 'csv', 'xlsx', 'xls', 'pdf'], 
            key="unified_upload",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            img_files = [f for f in uploaded_files if f.name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]
            data_files = [f for f in uploaded_files if f.name.split('.')[-1].lower() in ['csv', 'xlsx', 'xls', 'pdf']]

            col_a, col_b = st.columns(2)
            
            # --- 批量处理数据文件 (AI 文本解析) ---
            if data_files:
                with col_a:
                    st.info(f"检测到 {len(data_files)} 个数据文件")
                    if st.button(f"AI 智能解析导入", key="btn_import_batch"):
                        if not sf_api_key:
                            st.error("请先配置 SILICONFLOW_API_KEY")
                        else:
                            total_added = 0
                            total_skipped = 0
                            
                            with st.spinner("正在提取文本并呼叫 DeepSeek 进行分析 (可能需要几十秒)..."):
                                batch_df = pd.DataFrame()
                                
                                for f in data_files:
                                    # 注意：这里需要传入 api_key
                                    df_new, err = BillParser.identify_and_parse(f, sf_api_key)
                                    if err:
                                        st.error(f"文件 {f.name} 解析失败: {err}")
                                    elif df_new is not None and not df_new.empty:
                                        batch_df = pd.concat([batch_df, df_new], ignore_index=True)
                                
                                if not batch_df.empty:
                                    merged_df, added_count, skipped_count = BillParser.merge_and_deduplicate(
                                        st.session_state.ledger_data, batch_df
                                    )
                                    total_added += added_count
                                    total_skipped += skipped_count
                                    
                                    if total_added > 0:
                                        if dm.save_data(merged_df, st.session_state.get('github_sha')):
                                            st.session_state.ledger_data = merged_df
                                            st.session_state.github_sha = dm.load_data()[1]
                                            st.success(f"🎉 成功！DeepSeek 帮你提取了 {total_added} 条新记录。")
                                            if total_skipped > 0:
                                                st.info(f"🛡️ 自动跳过了 {total_skipped} 条重复记录")
                                            st.rerun()
                                        else:
                                            st.error("保存失败")
                                    else:
                                        st.warning(f"分析完成，但所有记录均已存在 (跳过 {total_skipped} 条)。")
                                else:
                                    st.warning("AI 没有发现有效的交易数据，可能是文件内容为空或格式过于特殊。")

            # --- 批量/单张 图片处理 (OCR) ---
            if img_files:
                with col_b:
                    st.info(f"检测到 {len(img_files)} 张图片")
                    if 'ocr_queue' not in st.session_state:
                        st.session_state.ocr_queue = []
                        
                    if st.button(f"开始 AI 视觉识别", key="btn_ocr_batch"):
                        if not sf_api_key:
                            st.error("请配置 SILICONFLOW_API_KEY")
                        else:
                            with st.spinner("AI 正在逐张读取..."):
                                for img_f in img_files:
                                    data, err = process_bill_image(img_f, sf_api_key)
                                    if not err and data:
                                        data['_filename'] = img_f.name
                                        st.session_state.ocr_queue.append(data)
                                    else:
                                        st.error(f"{img_f.name} 识别失败: {err}")
                            st.rerun()

        # --- OCR 结果确认队列 ---
        if 'ocr_queue' in st.session_state and len(st.session_state.ocr_queue) > 0:
            st.divider()
            st.subheader(f"🔍 待确认 OCR 结果 (剩余 {len(st.session_state.ocr_queue)} 个)")
            
            current_ocr = st.session_state.ocr_queue[0]
            
            with st.container(border=True):
                st.caption(f"来源文件: {current_ocr.get('_filename', 'Unknown')}")
                with st.form("ocr_confirm_queue"):
                    c1, c2 = st.columns(2)
                    o_date = c1.date_input("日期", pd.to_datetime(current_ocr.get('date', str(date.today()))))
                    o_type = c2.selectbox("类型", ["支出", "收入"], index=1 if current_ocr.get('type') == '收入' else 0)
                    o_amt = c1.number_input("金额", float(current_ocr.get('amount', 0)))
                    o_cat = c2.text_input("分类", current_ocr.get('category', '餐饮'))
                    o_desc = st.text_input("备注", current_ocr.get('merchant', ''))
                    
                    col_submit, col_skip = st.columns([1, 1])
                    if col_submit.form_submit_button("✅ 确认添加"):
                        new_row = {"日期": str(o_date), "类型": o_type, "金额": o_amt, "备注": o_desc, "分类": o_cat}
                        st.session_state.ledger_data = pd.concat([st.session_state.ledger_data, pd.DataFrame([new_row])], ignore_index=True)
                        dm.save_data(st.session_state.ledger_data, st.session_state.get('github_sha'))
                        st.session_state.github_sha = dm.load_data()[1]
                        st.session_state.ocr_queue.pop(0)
                        st.rerun()
                        
                    if col_skip.form_submit_button("🗑️ 跳过此条"):
                        st.session_state.ocr_queue.pop(0)
                        st.rerun()

    # --- Manual Tab ---
    with tab_manual:
        with st.form("manual_form"):
            c_m1, c_m2 = st.columns(2)
            m_date = c_m1.date_input("日期", date.today())
            m_type = c_m2.selectbox("类型", ["支出", "收入"])
            m_amt = c_m1.number_input("金额", step=1.0)
            m_cat = c_m2.selectbox("分类", ["餐饮", "交通", "购物", "居住", "娱乐", "工资", "其他"])
            m_desc = st.text_input("备注")
            
            if st.form_submit_button("💾 保存记录"):
                new_row = {"日期": str(m_date), "类型": m_type, "金额": m_amt, "备注": m_desc, "分类": m_cat}
                st.session_state.ledger_data = pd.concat([st.session_state.ledger_data, pd.DataFrame([new_row])], ignore_index=True)
                dm.save_data(st.session_state.ledger_data, st.session_state.get('github_sha'))
                st.session_state.github_sha = dm.load_data()[1]
                st.rerun()

    st.divider()

    # 5. 历史账单 & 可视化
    if not st.session_state.ledger_data.empty:
        st.subheader("📊 历史账单")
        edited_df = st.data_editor(
            st.session_state.ledger_data,
            num_rows="dynamic",
            use_container_width=True,
            key="history_editor"
        )
        if st.button("🔄 同步表格修改"):
            if dm.save_data(edited_df, st.session_state.get('github_sha')):
                st.session_state.ledger_data = edited_df
                st.session_state.github_sha = dm.load_data()[1]
                st.success("同步成功")
                st.rerun()
        
        st.subheader("📈 消费透视")
        chart_df = st.session_state.ledger_data.copy()
        chart_df['金额'] = pd.to_numeric(chart_df['金额'], errors='coerce').fillna(0)
        chart_df['日期'] = pd.to_datetime(chart_df['日期']).dt.date
        expense_df = chart_df[chart_df['类型'] == '支出']
        
        if not expense_df.empty:
            t1, t2 = st.tabs(["📊 分类占比", "📉 每日趋势"])
            with t1:
                st.bar_chart(expense_df.groupby('分类')['金额'].sum().sort_values(ascending=False), color="#FF4B4B")
            with t2:
                st.line_chart(expense_df.groupby('日期')['金额'].sum())
    else:
        st.info("暂无数据")

    # 6. AI 分析
    with st.expander("🤖 AI 财务分析"):
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
