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

# --- 账单解析与去重类 ---
class BillParser:
    @staticmethod
    def identify_and_parse(file):
        """智能识别文件类型并解析"""
        filename = file.name.lower()
        
        if filename.endswith('.csv'):
            return BillParser._parse_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            return BillParser._parse_excel(file)
        else:
            return None, "不支持的文件格式，请上传 CSV 或 Excel"

    @staticmethod
    def _parse_csv(file):
        """解析 CSV (微信/支付宝)"""
        try:
            content = file.getvalue().decode('utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            content = file.getvalue().decode('gbk', errors='ignore')

        # 简单的特征检测
        if "微信支付账单明细" in content or "交易时间" in content:
            return BillParser._parse_wechat_content(content)
        elif "支付宝交易记录明细" in content or "Partner Transaction ID" in content or "交易创建时间" in content:
            # 支付宝格式较多，尝试通用解析
            return BillParser._parse_alipay_content(content)
        elif "招商银行" in content:
            # 极少见招行导出CSV，但防万一
            return None, "请上传招商银行的 Excel (xls/xlsx) 格式文件"
        else:
            # 尝试盲解
            return BillParser._parse_alipay_content(content)

    @staticmethod
    def _parse_excel(file):
        """解析 Excel (招商银行等)"""
        try:
            df = pd.read_excel(file)
        except Exception as e:
            return None, f"Excel 读取失败: {e}"

        # 招商银行特征检测
        # 招行表头通常包含: 交易日期, 交易时间, 支出, 收入, 余额, 交易类型, 交易备注
        # 或者: 记账日期, 货币, 交易金额, 联机余额, 交易摘要
        cols = [str(c) for c in df.columns]
        col_str = " ".join(cols)
        
        if "交易日期" in col_str and ("支出" in col_str or "交易金额" in col_str):
            return BillParser._parse_cmb(df)
        
        return None, "未识别的 Excel 账单格式，目前仅优化支持招商银行。"

    @staticmethod
    def _parse_wechat_content(content):
        lines = content.split('\n')
        start_row = 0
        for i, line in enumerate(lines):
            if "交易时间" in line:
                start_row = i
                break
        
        try:
            df = pd.read_csv(StringIO(content), header=start_row)
        except:
            return None, "微信账单解析失败"

        df.columns = [c.strip() for c in df.columns]
        df = df[df['当前状态'] == '支付成功']
        
        results = []
        for _, row in df.iterrows():
            row_type = row['收/支']
            if row_type == "/" or row_type == "不计收支": continue
            
            final_type = "支出" if row_type == "支出" else "收入"
            amt = float(str(row['金额(元)']).replace('¥', '').replace(',', ''))
            
            try:
                d_str = pd.to_datetime(row['交易时间']).strftime('%Y-%m-%d')
            except:
                continue

            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": amt,
                "备注": f"{row['交易对方']} - {row['商品']}",
                "分类": "微信导入"
            })
        return pd.DataFrame(results), None

    @staticmethod
    def _parse_alipay_content(content):
        # 支付宝处理逻辑
        lines = content.split('\n')
        start_row = 0
        for i, line in enumerate(lines):
            if "交易时间" in line and "交易对方" in line:
                start_row = i
                break
        
        try:
            df = pd.read_csv(StringIO(content), header=start_row)
        except:
            return None, "支付宝账单解析失败"

        df.columns = [c.strip() for c in df.columns]
        if '交易状态' in df.columns:
            df = df[df['交易状态'].isin(['交易成功', '支付成功', '已支出'])]

        results = []
        for _, row in df.iterrows():
            if '金额' not in row or pd.isna(row['金额']): continue
            row_type = str(row.get('收/支', '')).strip()
            if row_type == "不计收支" or row_type == "": continue
            
            final_type = "支出" if row_type == "支出" else "收入"
            amt = float(str(row['金额']))
            
            try:
                d_str = pd.to_datetime(row['交易时间']).strftime('%Y-%m-%d')
            except:
                continue

            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": amt,
                "备注": f"{row.get('交易对方','')} {row.get('商品说明','')}".strip(),
                "分类": row.get('交易分类', '支付宝导入')
            })
        return pd.DataFrame(results), None

    @staticmethod
    def _parse_cmb(df):
        """招商银行 Excel 解析逻辑"""
        # 寻找表头行
        header_row_idx = 0
        for i in range(len(df)):
            row_vals = [str(v) for v in df.iloc[i].values]
            if "交易日期" in row_vals or "记账日期" in row_vals:
                header_row_idx = i
                break
        
        # 重新读取，指定header
        df.columns = df.iloc[header_row_idx]
        df = df.iloc[header_row_idx+1:]
        df.columns = [str(c).strip() for c in df.columns]
        
        results = []
        for _, row in df.iterrows():
            # 招行格式可能有多种，常见一种：交易日期, 支出, 收入, 交易备注
            date_val = row.get('交易日期') or row.get('记账日期')
            if pd.isna(date_val): continue
            
            # 格式化日期
            try:
                # 招行日期可能是 20230101 或 2023-01-01
                d_str = pd.to_datetime(str(date_val)).strftime('%Y-%m-%d')
            except:
                continue

            # 金额处理
            expense = row.get('支出', 0)
            income = row.get('收入', 0)
            # 有些版本是“交易金额”带负号
            trans_amt = row.get('交易金额', 0)

            final_amt = 0.0
            final_type = "支出"
            
            if trans_amt != 0:
                trans_amt = float(str(trans_amt).replace(',', ''))
                final_amt = abs(trans_amt)
                final_type = "支出" if trans_amt < 0 else "收入"
            elif expense and float(str(expense).replace(',', '')) > 0:
                final_amt = float(str(expense).replace(',', ''))
                final_type = "支出"
            elif income and float(str(income).replace(',', '')) > 0:
                final_amt = float(str(income).replace(',', ''))
                final_type = "收入"
            else:
                continue # 金额为0跳过

            memo = str(row.get('交易备注') or row.get('交易摘要') or "")
            
            results.append({
                "日期": d_str,
                "类型": final_type,
                "金额": final_amt,
                "备注": memo.strip(),
                "分类": "招行导入"
            })
            
        return pd.DataFrame(results), None

    @staticmethod
    def merge_and_deduplicate(old_df, new_df):
        """
        合并并去重
        策略：如果 Date + Amount + Type 相同，视为重复。
        针对银行账单的特殊处理：如果备注包含 '支付宝'/'微信' 且金额重复，更要跳过。
        """
        if new_df is None or new_df.empty:
            return old_df, 0, 0

        added_rows = []
        skipped_count = 0
        
        # 建立索引以加速查找 (日期+金额+类型)
        # 为避免浮点数精度问题，金额保留2位小数的字符串作为Key
        existing_keys = set()
        for _, row in old_df.iterrows():
            key = f"{row['日期']}_{float(row['金额']):.2f}_{row['类型']}"
            existing_keys.add(key)

        for _, row in new_df.iterrows():
            amt = float(row['金额'])
            key = f"{row['日期']}_{amt:.2f}_{row['类型']}"
            
            if key in existing_keys:
                # 发现潜在重复
                # 如果是银行账单，且包含第三方支付关键字，这是典型的“重合账单”，必须跳过
                memo = str(row['备注'])
                if "招行" in str(row.get('分类', '')):
                    if any(k in memo for k in ["支付宝", "微信", "财付通", "Tenpay", "Alipay"]):
                        skipped_count += 1
                        continue
                
                # 即使没有关键字，只要日期金额完全一致，也视为重复（用户不希望重复记录）
                skipped_count += 1
                continue
            else:
                added_rows.append(row)
                existing_keys.add(key) # 避免新文件内部自我重复

        if not added_rows:
            return old_df, 0, skipped_count
            
        return pd.concat([old_df, pd.DataFrame(added_rows)], ignore_index=True), len(added_rows), skipped_count

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
    tab_auto, tab_manual = st.tabs(["📤 智能导入 (图片/文件)", "✍️ 手动记账"])

    with tab_auto:
        st.markdown("""
        <small>支持格式：
        1. **图片** (jpg/png) -> AI 自动识别
        2. **文件** (csv/xlsx/xls) -> 微信/支付宝/招商银行账单导入 (自动去重)
        </small>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("点击上传账单或截图", type=['png', 'jpg', 'jpeg', 'csv', 'xlsx', 'xls'], key="unified_upload")
        
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # --- 分支 A: 图片处理 (OCR) ---
            if file_type in ['png', 'jpg', 'jpeg']:
                if st.button("开始 AI 识别", key="btn_ocr"):
                    if not sf_api_key:
                        st.error("请配置 SILICONFLOW_API_KEY")
                    else:
                        with st.spinner("AI 正在读取账单..."):
                            data, err = process_bill_image(uploaded_file, sf_api_key)
                            if err:
                                st.error(err)
                            else:
                                st.session_state.temp_ocr_data = data
                
                # OCR 结果确认框
                if 'temp_ocr_data' in st.session_state:
                    res = st.session_state.temp_ocr_data
                    with st.form("ocr_confirm"):
                        st.write("🔍 识别结果确认：")
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

            # --- 分支 B: 文件导入 (CSV/Excel) ---
            elif file_type in ['csv', 'xlsx', 'xls']:
                if st.button("解析并导入文件", key="btn_import"):
                    with st.spinner("正在解析..."):
                        df_new, err = BillParser.identify_and_parse(uploaded_file)
                        
                        if err:
                            st.error(err)
                        elif df_new is not None and not df_new.empty:
                            # 执行合并与去重
                            merged_df, added_count, skipped_count = BillParser.merge_and_deduplicate(
                                st.session_state.ledger_data, df_new
                            )
                            
                            if added_count > 0:
                                if dm.save_data(merged_df, st.session_state.get('github_sha')):
                                    st.session_state.ledger_data = merged_df
                                    st.session_state.github_sha = dm.load_data()[1]
                                    st.success(f"🎉 成功导入 {added_count} 条记录！")
                                    if skipped_count > 0:
                                        st.info(f"🛡️ 自动跳过了 {skipped_count} 条重复记录 (包含微信/支付宝与招行重合部分)")
                                    st.rerun()
                                else:
                                    st.error("保存失败")
                            else:
                                st.warning(f"未添加任何记录。检测到 {skipped_count} 条重复数据。")

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
                        r = requests.post("[https://api.siliconflow.cn/v1/chat/completions](https://api.siliconflow.cn/v1/chat/completions)", 
                                        headers={"Authorization": f"Bearer {sf_api_key}"}, json=payload)
                        st.markdown(r.json()['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error(f"AI 服务异常: {e}")

if __name__ == "__main__":
    main()
