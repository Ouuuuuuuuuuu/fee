import streamlit as st
import pandas as pd
import datetime
from datetime import date
import requests
import json
import base64
from io import StringIO, BytesIO
import os

# --- 页面配置 ---
st.set_page_config(page_title="AI 智能账本", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
# 推荐的视觉模型，SiliconFlow 上可用
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
# 文本分析模型
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

# --- 账单解析类 ---
class BillParser:
    @staticmethod
    def parse_wechat(file):
        """解析微信账单 CSV"""
        try:
            content = file.getvalue().decode('utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            content = file.getvalue().decode('gbk', errors='ignore')

        lines = content.split('\n')
        start_row = 0
        for i, line in enumerate(lines):
            if "交易时间" in line:
                start_row = i
                break
        
        if start_row == 0 and "交易时间" not in lines[0]:
             return None, "未找到微信账单表头，请确认文件格式"

        try:
            df = pd.read_csv(StringIO(content), header=start_row)
        except Exception as e:
            return None, f"CSV解析失败: {str(e)}"

        # 微信字段清洗
        df.columns = [c.strip() for c in df.columns]
        required_cols = ['交易时间', '金额(元)', '收/支', '交易对方', '商品', '当前状态']
        
        if not all(col in df.columns for col in required_cols):
             return None, f"列名不匹配，检测到的列: {list(df.columns)}"

        df = df[df['当前状态'] == '支付成功']
        
        results = []
        for _, row in df.iterrows():
            amt = float(str(row['金额(元)']).replace('¥', '').replace(',', ''))
            row_type = row['收/支']
            
            final_type = "支出" if row_type == "支出" else "收入"
            if row_type == "/" or row_type == "不计收支":
                continue

            try:
                d_str = pd.to_datetime(row['交易时间']).strftime('%Y-%m-%d')
            except:
                continue

            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": amt,
                "备注": f"{row['交易对方']} - {row['商品']}",
                "分类": "导入/未分类"
            })
            
        return pd.DataFrame(results), None

    @staticmethod
    def parse_alipay(file):
        """解析支付宝账单"""
        try:
            content = file.getvalue().decode('gbk')
        except UnicodeDecodeError:
            file.seek(0)
            content = file.getvalue().decode('utf-8', errors='ignore')

        lines = content.split('\n')
        start_row = 0
        for i, line in enumerate(lines):
            if "交易时间" in line and "交易对方" in line:
                start_row = i
                break
        
        try:
            df = pd.read_csv(StringIO(content), header=start_row, encoding='gbk')
        except:
             df = pd.read_csv(StringIO(content), header=start_row)

        df.columns = [c.strip() for c in df.columns]
        
        if '交易状态' in df.columns:
            df = df[df['交易状态'].isin(['交易成功', '支付成功', '已支出'])]

        results = []
        for _, row in df.iterrows():
            if '金额' not in row or pd.isna(row['金额']): continue

            amt = float(str(row['金额']))
            row_type = str(row.get('收/支', '')).strip()
            
            final_type = "支出" if row_type == "支出" else "收入"
            if row_type == "不计收支" or row_type == "":
                continue

            try:
                d_str = pd.to_datetime(row['交易时间']).strftime('%Y-%m-%d')
            except:
                continue
            
            cat = row.get('交易分类', '导入/未分类')
            merchant = row.get('交易对方', '')
            desc = row.get('商品说明', '')

            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": amt,
                "备注": f"{merchant} {desc}".strip(),
                "分类": cat
            })
            
        return pd.DataFrame(results), None

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
        # 修正 URL 格式问题
        response = requests.post(
            "[https://api.siliconflow.cn/v1/chat/completions](https://api.siliconflow.cn/v1/chat/completions)",
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
        st.sidebar.warning("📂 数据存储: 本地模式 (重启后Streamlit Cloud会重置数据)")

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

    # 4. 记账功能区
    tab_ocr, tab_manual, tab_import = st.tabs(["📸 截图记账 (OCR)", "✍️ 手动记账", "📂 导入账单(Excel/CSV)"])

    # --- Tab 1: OCR ---
    with tab_ocr:
        c1, c2 = st.columns([1, 1])
        with c1:
            # 修复: 明确设置 label 为 "上传截图"，防止出现 label 不能为空的警告
            uploaded_file = st.file_uploader("上传截图", type=['png', 'jpg', 'jpeg'], key="ocr_upload")
            if uploaded_file and st.button("开始识别", key="btn_ocr"):
                if not sf_api_key:
                    st.error("请先配置 SILICONFLOW_API_KEY")
                else:
                    with st.spinner("AI 正在提取信息..."):
                        data, err = process_bill_image(uploaded_file, sf_api_key)
                        if err:
                            st.error(err)
                        else:
                            st.success("识别成功！")
                            st.session_state.temp_ocr_data = data
        
        with c2:
            if 'temp_ocr_data' in st.session_state:
                res = st.session_state.temp_ocr_data
                with st.form("ocr_confirm"):
                    st.write("确认识别结果：")
                    o_date = st.date_input("日期", pd.to_datetime(res.get('date', str(date.today()))))
                    o_type = st.selectbox("类型", ["支出", "收入"], index=1 if res.get('type') == '收入' else 0)
                    o_amt = st.number_input("金额", float(res.get('amount', 0)))
                    o_cat = st.text_input("分类", res.get('category', '餐饮'))
                    o_desc = st.text_input("备注", res.get('merchant', ''))
                    
                    if st.form_submit_button("✅ 确认添加"):
                        new_row = {"日期": str(o_date), "类型": o_type, "金额": o_amt, "备注": o_desc, "分类": o_cat}
                        st.session_state.ledger_data = pd.concat([st.session_state.ledger_data, pd.DataFrame([new_row])], ignore_index=True)
                        dm.save_data(st.session_state.ledger_data, st.session_state.get('github_sha'))
                        st.session_state.github_sha = dm.load_data()[1]
                        del st.session_state.temp_ocr_data
                        st.rerun()

    # --- Tab 2: Manual ---
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

    # --- Tab 3: Import ---
    with tab_import:
        st.info("💡 提示：支持微信或支付宝导出的 CSV 文件。系统会自动忽略已存在的记录（日期、金额、类型、备注完全一致的）。")
        import_file = st.file_uploader("上传账单文件", type=['csv'], key="bill_import")
        
        if import_file:
            bill_type = st.radio("选择账单来源", ["微信", "支付宝"], horizontal=True)
            if st.button("开始解析并导入"):
                with st.spinner("正在解析文件..."):
                    if bill_type == "微信":
                        df_new, err = BillParser.parse_wechat(import_file)
                    else:
                        df_new, err = BillParser.parse_alipay(import_file)
                    
                    if err:
                        st.error(err)
                    elif df_new is not None and not df_new.empty:
                        # 1. 组合新旧数据
                        old_df = st.session_state.ledger_data.copy()
                        
                        # 2. 去重逻辑
                        combined = pd.concat([old_df, df_new], ignore_index=True)
                        deduplicated = combined.drop_duplicates(subset=['日期', '金额', '备注', '类型'], keep='first')
                        
                        # 3. 计算新增数量
                        added_count = len(deduplicated) - len(old_df)
                        ignored_count = len(df_new) - added_count
                        
                        if added_count > 0:
                            if dm.save_data(deduplicated, st.session_state.get('github_sha')):
                                st.session_state.ledger_data = deduplicated
                                st.session_state.github_sha = dm.load_data()[1]
                                st.success(f"🎉 成功导入 {added_count} 条新记录！")
                                if ignored_count > 0:
                                    st.warning(f"🛡️ 自动忽略了 {ignored_count} 条已存在的重复记录。")
                                st.rerun()
                            else:
                                st.error("保存失败")
                        else:
                            st.warning(f"所有 {len(df_new)} 条记录均已存在，无需更新。")
                    else:
                        st.warning("解析成功，但没有发现有效交易记录。")

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

        col_save, col_info = st.columns([1, 4])
        with col_save:
            if st.button("🔄 同步表格修改"):
                if dm.save_data(edited_df, st.session_state.get('github_sha')):
                    st.session_state.ledger_data = edited_df
                    st.session_state.github_sha = dm.load_data()[1]
                    st.success("同步成功")
                    st.rerun()
        
        st.divider()
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
                        # 修复 URL 格式问题
                        r = requests.post("[https://api.siliconflow.cn/v1/chat/completions](https://api.siliconflow.cn/v1/chat/completions)", 
                                        headers={"Authorization": f"Bearer {sf_api_key}"}, json=payload)
                        st.markdown(r.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"AI 服务异常: {e}")

if __name__ == "__main__":
    main()
