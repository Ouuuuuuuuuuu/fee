import streamlit as st
import pandas as pd
import datetime
from datetime import date
import requests
import json
import base64
from io import StringIO, BytesIO
import os
import fitz  # PyMuPDF
import re
from openai import OpenAI, APITimeoutError
import concurrent.futures
import time

# --- 页面配置 ---
st.set_page_config(page_title="AI 账本 ", page_icon="💰", layout="wide")

# --- 常量配置 ---
DEFAULT_TARGET_SPEND = 60.0  # 每日体面支出标准
GITHUB_API_URL = "https://api.github.com"
VISION_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
TEXT_MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"

# --- 缓存资源：获取 LLM 客户端 ---
@st.cache_resource
def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# --- 工具函数：鲁棒的 JSON 提取 ---
def extract_json_from_text(text):
    """使用正则从混合文本中提取 JSON 数组或对象"""
    text = text.replace("```json", "").replace("```", "").strip()
    
    # 尝试提取数组 [...]
    match_array = re.search(r'\[.*\]', text, re.DOTALL)
    if match_array:
        try:
            return json.loads(match_array.group())
        except:
            pass
            
    # 尝试提取对象 {...}
    match_obj = re.search(r'\{.*\}', text, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group())
        except:
            pass

    # 最后的手段：尝试直接解析
    try:
        return json.loads(text)
    except:
        return None

# --- 存储类 ---
class DataManager:
    """数据管理类，支持 GitHub 远程存储和本地 CSV 存储"""
    def __init__(self, github_token=None, repo=None, filename="ledger.csv"):
        self.github_token = github_token
        # 兼容完整 URL 或 repo path
        if repo and repo.startswith("http"):
            self.repo = repo.rstrip("/").split("github.com/")[-1]
        else:
            self.repo = repo
        self.filename = filename
        self.use_github = bool(github_token and self.repo)

    def load_data(self):
        """加载数据，返回 DataFrame 和 SHA"""
        if self.use_github:
            return self._load_from_github()
        else:
            return self._load_from_local()

    def save_data(self, df, sha=None):
        """保存数据"""
        # 确保数据格式统一
        if '日期' in df.columns:
            df['日期'] = df['日期'].astype(str)
        if self.use_github:
            return self._save_to_github(df, sha)
        else:
            return self._save_to_local(df)

    def _load_from_local(self):
        if os.path.exists(self.filename):
            try:
                return pd.read_csv(self.filename), None
            except:
                pass
        return self._create_empty_df(), None

    def _save_to_local(self, df):
        df.to_csv(self.filename, index=False)
        return True

    # --- 优化：添加缓存，避免每次刷新页面都请求 GitHub，缓存 5 分钟 ---
    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_github_content(_self):
        """内部函数：实际执行网络请求，单独拆分以支持缓存"""
        headers = {
            "Authorization": f"token {_self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        url = f"{GITHUB_API_URL}/repos/{_self.repo}/contents/{_self.filename}"
        try:
            # timeout 保持 60s
            response = requests.get(url, headers=headers, timeout=60)
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 404:
                return None, 404
            else:
                return None, response.status_code
        except Exception as e:
            return None, str(e)

    def _load_from_github(self):
        # 调用带缓存的读取函数
        content, error = self._fetch_github_content()
        
        if content:
            csv_str = base64.b64decode(content['content']).decode('utf-8')
            try:
                df = pd.read_csv(StringIO(csv_str))
                expected_df = self._create_empty_df()
                for col in expected_df.columns:
                    if col not in df.columns:
                        df[col] = ""
                return df, content['sha']
            except pd.errors.EmptyDataError:
                return self._create_empty_df(), content['sha']
        elif error == 404:
            return self._create_empty_df(), None
        else:
            if error:
                st.error(f"GitHub 读取错误: {error}")
            return self._create_empty_df(), None

    def _save_to_github(self, df, sha):
        start_time = time.time()
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        csv_str = df.to_csv(index=False)
        content_bytes = base64.b64encode(csv_str.encode('utf-8')).decode('utf-8')
        
        url = f"{GITHUB_API_URL}/repos/{self.repo}/contents/{self.filename}"
        data = {
            "message": f"Update ledger {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "content": content_bytes
        }
        if sha:
            data["sha"] = sha
            
        try:
            # 增加 timeout 到 60秒
            response = requests.put(url, headers=headers, data=json.dumps(data), timeout=60)
            end_time = time.time()
            if st.session_state.get('debug_mode'):
                st.toast(f"☁️ GitHub 保存耗时: {end_time - start_time:.2f}s")
            
            if response.status_code in [200, 201]:
                # --- 优化：保存成功后清除读取缓存，确保下次读取是新的 ---
                self._fetch_github_content.clear()
                return True
            else:
                st.error(f"GitHub 保存失败: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            st.error(f"保存时网络异常: {e}")
            return False

    @staticmethod
    def _create_empty_df():
        return pd.DataFrame(columns=["日期", "类型", "金额", "备注", "分类"])

# --- 智能账单解析类 ---
class BillParser:
    @staticmethod
    def identify_and_parse(filename, file_bytes, api_key):
        """
        处理单个文件内容
        注意：这里不再接收 Streamlit 的 UploadedFile 对象，而是接收 (filename, file_bytes)
        从而彻底解决 'missing ScriptRunContext' 问题
        """
        t_start = time.time()
        debug_info = {}
        
        if not api_key:
            return None, "未配置 API Key", {}

        filename = filename.lower()
        content_text = ""
        source_type = "未知文件"
        
        try:
            # 1. 提取文本 (基于 file_bytes)
            t_read_start = time.time()
            
            # 使用 BytesIO 包装二进制数据，使其像文件一样可读
            file_stream = BytesIO(file_bytes)
            
            if filename.endswith('.csv'):
                source_type = "CSV账单"
                try:
                    content_text = file_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    content_text = file_bytes.decode('gbk', errors='ignore')
            
            elif filename.endswith(('.xls', '.xlsx')):
                source_type = "Excel账单"
                try:
                    xls = pd.read_excel(file_stream, sheet_name=None)
                    text_parts = []
                    for sheet_name, df in xls.items():
                        text_parts.append(f"--- Sheet: {sheet_name} ---\n")
                        text_parts.append(df.to_csv(index=False))
                    content_text = "\n".join(text_parts)
                except Exception as e:
                    return None, f"Excel 读取失败: {e}", debug_info

            elif filename.endswith('.pdf'):
                source_type = "PDF账单"
                try:
                    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                        text_parts = [page.get_text() for page in doc]
                        content_text = "\n".join(text_parts)
                except Exception as e:
                    return None, f"PDF 读取失败: {e}", debug_info
            else:
                return None, "不支持的文件格式", {}

            debug_info['read_time'] = time.time() - t_read_start
            debug_info['text_length'] = len(content_text)

            if not content_text.strip():
                return None, "无法提取文本内容", debug_info
            
            # 2. AI 解析
            res_df, err, ai_debug = BillParser._call_ai_parser(content_text, source_type, api_key)
            debug_info.update(ai_debug)
            
            debug_info['total_time'] = time.time() - t_start
            return res_df, err, debug_info

        except Exception as e:
            return None, f"文件解析错误: {str(e)}", debug_info

    @staticmethod
    def _call_ai_parser(content_text, source_type, api_key):
        debug_info = {}
        t_ai_start = time.time()
        
        # 强化 Prompt
        system_prompt = """
        你是一个严谨的财务数据提取专家。请从文本中提取交易流水。
        核心原则：宁缺毋假。绝对禁止捏造、模拟或推测数据。只提取文本中明确存在的交易。
        
        规则：
        1. 返回标准 JSON 数组 `[{"date": "YYYY-MM-DD", "type": "支出/收入", "amount": 10.5, "merchant": "商户名", "category": "分类"}, ...]`
        2. category 从以下选取：[餐饮, 交通, 购物, 居住, 娱乐, 工资, 理财, 医疗, 其他]。
        3. 仅提取真实交易，忽略余额、表头。
        4. amount 必须为正数 (float)。
        5. 如果没有找到有效交易，必须返回 []，不要编造。
        """

        user_prompt = f"请处理这份 {source_type}，当前年份默认为 {datetime.datetime.now().year}。\n数据内容如下:\n{content_text}"

        client = get_llm_client(api_key)
        try:
            response = client.chat.completions.create(
                model=TEXT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=8192,
                temperature=0.0
            )
            debug_info['ai_latency'] = time.time() - t_ai_start
            
            t_parse_start = time.time()
            ai_content = response.choices[0].message.content
            data_list = extract_json_from_text(ai_content)
            debug_info['parse_time'] = time.time() - t_parse_start

            if data_list is None or not isinstance(data_list, list):
                return None, f"AI 返回格式无法解析: {ai_content[:100]}...", debug_info
            
            if not data_list:
                return None, "未提取到有效数据", debug_info

            df = pd.DataFrame(data_list)
            
            # 标准化列名
            col_map = {"date": "日期", "type": "类型", "amount": "金额", "merchant": "备注", "category": "分类"}
            df = df.rename(columns=col_map)
            
            # 补全缺失列
            for col in col_map.values():
                if col not in df.columns:
                    df[col] = ""

            # 数据清洗
            df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
            df['分类'] = df['分类'].fillna("其他")
            # 简单日期清洗
            df['日期'] = df['日期'].apply(lambda x: str(x).split(' ')[0])

            return df, None, debug_info

        except Exception as e:
            return None, f"AI 请求失败: {str(e)}", debug_info

    @staticmethod
    def merge_and_deduplicate(old_df, new_df):
        """合并并去重"""
        if new_df is None or new_df.empty:
            return old_df, 0, 0

        # 构造指纹列
        def make_fingerprint(df):
            return df['日期'].astype(str) + "_" + \
                   df['金额'].astype(float).round(2).astype(str) + "_" + \
                   df['类型'] + "_" + \
                   df['备注'].str.slice(0, 5)

        if old_df.empty:
            return new_df, len(new_df), 0

        old_df['_fp'] = make_fingerprint(old_df)
        new_df['_fp'] = make_fingerprint(new_df)
        
        existing_fps = set(old_df['_fp'].tolist())
        
        # 筛选新行
        to_add = new_df[~new_df['_fp'].isin(existing_fps)].copy()
        skipped_count = len(new_df) - len(to_add)
        
        # 清理临时列
        if '_fp' in old_df.columns: del old_df['_fp']
        if '_fp' in to_add.columns: del to_add['_fp']
        
        final_df = pd.concat([old_df, to_add], ignore_index=True)
        # 按日期降序排序
        final_df = final_df.sort_values(by="日期", ascending=False).reset_index(drop=True)
        
        return final_df, len(to_add), skipped_count

# --- 图像处理 ---
def process_bill_image(filename, image_bytes, api_key):
    """
    处理单个图片
    同样不再接收 UploadedFile，而是接收 (filename, image_bytes)
    """
    if not api_key: return None, "未配置 API Key", {}
    
    t_start = time.time()
    debug_info = {}
    
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        client = get_llm_client(api_key)
        prompt = "提取账单信息。返回JSON: {date: 'YYYY-MM-DD', amount: float, merchant: string, category: string, type: '支出'|'收入'}。"

        t_ai_start = time.time()
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=2048
        )
        debug_info['ai_latency'] = time.time() - t_ai_start
        
        content = response.choices[0].message.content
        data = extract_json_from_text(content)
        
        debug_info['total_time'] = time.time() - t_start
        
        if isinstance(data, list): data = data[0]
        if not data: return None, "无法识别图片内容", debug_info
        
        return data, None, debug_info
    except Exception as e:
        return None, f"视觉识别错误: {e}", debug_info

# --- 主程序 ---
def main():
    # 1. 侧边栏配置
    st.sidebar.title("⚙️ 财务设置")
    
    # --- 调试模式开关 ---
    st.session_state.debug_mode = st.sidebar.checkbox("🛠️ 开启调试模式", value=False)
    
    sf_api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    if not sf_api_key:
        sf_api_key = st.sidebar.text_input("SiliconFlow API Key", type="password")

    github_token = st.secrets.get("GITHUB_TOKEN", "")
    github_repo = st.secrets.get("GITHUB_REPO", "")

    dm = DataManager(github_token, github_repo)
    
    if dm.use_github:
        st.sidebar.success(f"☁️ 已连接 GitHub: {dm.repo}")
    else:
        st.sidebar.info("📂 使用本地存储 (刷新页面后数据可能丢失)")

    payday = st.sidebar.number_input("📅 每月发薪日", 1, 31, 10)
    current_cash = st.sidebar.number_input("💳 当前资产余额", value=3000.0)

    # 2. 初始化 Session State
    if 'ledger_data' not in st.session_state:
        # 这里只会在第一次加载时调用 load_data，或者缓存失效时
        with st.spinner("正在加载账本数据..."):
            df, sha = dm.load_data()
            st.session_state.ledger_data = df
            st.session_state.github_sha = sha

    # 3. 顶部概览
    st.title("💰 AI 智能账本 Pro")
    
    today = date.today()
    if today.day >= payday:
        target_date = date(today.year + (1 if today.month == 12 else 0), 1 if today.month == 12 else today.month + 1, payday)
    else:
        target_date = date(today.year, today.month, payday)
    days_left = (target_date - today).days

    current_month_str = today.strftime("%Y-%m")
    df_current = st.session_state.ledger_data.copy()
    if not df_current.empty:
        df_current['tmp_date'] = pd.to_datetime(df_current['日期'], errors='coerce')
        mask = (df_current['tmp_date'].dt.strftime('%Y-%m') == current_month_str) & (df_current['类型'] == '支出')
        month_spend = df_current.loc[mask, '金额'].sum()
    else:
        month_spend = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前资产", f"¥{current_cash:,.2f}")
    col2.metric("本月已支出", f"¥{month_spend:,.2f}")
    col3.metric("距离发薪", f"{days_left} 天")
    
    if days_left > 0:
        daily_budget = current_cash / days_left
        col4.metric("每日可用预算", f"¥{daily_budget:.0f}", 
                    delta=f"{daily_budget - DEFAULT_TARGET_SPEND:.0f}", 
                    delta_color="normal")

    st.divider()

    # 4. 核心功能区
    tab_import, tab_manual, tab_analysis = st.tabs(["📥 智能导入", "✍️ 手动记账", "📊 报表与AI"])

    with tab_import:
        uploaded_files = st.file_uploader("上传账单文件 (PDF/Excel/CSV) 或 票据图片", 
                                        accept_multiple_files=True,
                                        type=['png', 'jpg', 'csv', 'xlsx', 'pdf'])
        
        if uploaded_files:
            if st.button("🚀 开始 AI 识别", type="primary"):
                if not sf_api_key:
                    st.error("请先填写 API Key")
                    st.stop()

                doc_files = [f for f in uploaded_files if f.name.split('.')[-1].lower() in ['csv', 'xlsx', 'xls', 'pdf']]
                img_files = [f for f in uploaded_files if f.name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]

                batch_new_data = pd.DataFrame()
                
                # A. 处理文档 - 提高并发
                if doc_files:
                    st.caption(f"📄 正在并发分析 {len(doc_files)} 个文档...")
                    progress_bar = st.progress(0)
                    
                    # 关键修改：在主线程读取文件内容，只传递纯数据给子线程
                    # 这彻底解决了 ThreadPoolExecutor 中的 Streamlit 上下文丢失问题
                    doc_tasks = []
                    for f in doc_files:
                        doc_tasks.append({
                            "file_obj": f,             # 仅用于UI显示名字
                            "filename": f.name,        # 纯字符串
                            "bytes": f.getvalue()      # 纯二进制数据
                        })

                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        # 提交任务时，只传 name 和 bytes
                        future_map = {
                            executor.submit(BillParser.identify_and_parse, task["filename"], task["bytes"], sf_api_key): task["file_obj"] 
                            for task in doc_tasks
                        }
                        
                        for i, future in enumerate(concurrent.futures.as_completed(future_map)):
                            f_obj = future_map[future]
                            # 获取 debug_info
                            res, err, dbg = future.result()
                            
                            if st.session_state.debug_mode:
                                with st.expander(f"🔧 调试: {f_obj.name}", expanded=True): # 展开方便查看
                                    st.json(dbg)
                            
                            if res is not None and not res.empty:
                                batch_new_data = pd.concat([batch_new_data, res], ignore_index=True)
                                st.toast(f"✅ {f_obj.name} 解析成功")
                            else:
                                st.error(f"❌ {f_obj.name}: {err}")
                            progress_bar.progress((i + 1) / len(doc_files))

                # B. 处理图片 - 并行化
                if img_files:
                    st.caption(f"🖼️ 正在并发识别 {len(img_files)} 张图片...")
                    img_progress = st.progress(0)
                    
                    # 关键修改：图片也一样，主线程读取
                    img_tasks = []
                    for img in img_files:
                        img_tasks.append({
                            "file_obj": img,
                            "filename": img.name,
                            "bytes": img.getvalue()
                        })

                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        future_map = {
                            executor.submit(process_bill_image, task["filename"], task["bytes"], sf_api_key): task["file_obj"]
                            for task in img_tasks
                        }
                        
                        for i, future in enumerate(concurrent.futures.as_completed(future_map)):
                            img_obj = future_map[future]
                            res, err, dbg = future.result()
                            
                            if st.session_state.debug_mode:
                                with st.expander(f"🔧 调试: {img_obj.name}", expanded=True):
                                    st.json(dbg)

                            if res:
                                row = {
                                    "日期": res.get('date', str(date.today())),
                                    "类型": res.get('type', '支出'),
                                    "金额": res.get('amount', 0),
                                    "分类": res.get('category', '其他'),
                                    "备注": res.get('merchant', '图片识别')
                                }
                                batch_new_data = pd.concat([batch_new_data, pd.DataFrame([row])], ignore_index=True)
                                st.toast(f"✅ {img_obj.name} 识别成功")
                            else:
                                st.error(f"❌ {img_obj.name}: {err}")
                            img_progress.progress((i + 1) / len(img_files))

                # C. 合并入库
                if not batch_new_data.empty:
                    merged, added, skipped = BillParser.merge_and_deduplicate(st.session_state.ledger_data, batch_new_data)
                    if added > 0:
                        if dm.save_data(merged, st.session_state.get('github_sha')):
                            st.session_state.ledger_data = merged
                            st.session_state.github_sha = dm.load_data()[1] # 更新 sha
                            st.balloons()
                            st.success(f"成功导入 {added} 条记录 (自动去重 {skipped} 条)")
                    else:
                        st.warning(f"所有记录均已存在 (去重 {skipped} 条)")
                else:
                    st.warning("未能提取到有效数据")

    with tab_manual:
        with st.form("add_transaction"):
            c1, c2, c3 = st.columns(3)
            new_date = c1.date_input("日期", value=date.today())
            new_type = c2.selectbox("类型", ["支出", "收入"])
            new_amt = c3.number_input("金额", min_value=0.01, step=1.0)
            
            c4, c5 = st.columns([1, 2])
            new_cat = c4.selectbox("分类", ["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"])
            new_desc = c5.text_input("备注/商户")

            if st.form_submit_button("➕ 添加记录", use_container_width=True):
                new_row = pd.DataFrame([{
                    "日期": str(new_date), "类型": new_type, "金额": new_amt,
                    "分类": new_cat, "备注": new_desc
                }])
                merged, _, _ = BillParser.merge_and_deduplicate(st.session_state.ledger_data, new_row)
                if dm.save_data(merged, st.session_state.get('github_sha')):
                    st.session_state.ledger_data = merged
                    st.session_state.github_sha = dm.load_data()[1]
                    st.success("添加成功！")
                    st.rerun()

    with tab_analysis:
        if st.session_state.ledger_data.empty:
            st.info("暂无数据，请先记账。")
        else:
            st.subheader("📝 账单明细")
            
            edited_df = st.data_editor(
                st.session_state.ledger_data,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "日期": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
                    "金额": st.column_config.NumberColumn("金额", format="¥%.2f"),
                    "类型": st.column_config.SelectboxColumn("类型", options=["支出", "收入"]),
                    "分类": st.column_config.SelectboxColumn("分类", options=["餐饮", "交通", "购物", "居住", "娱乐", "医疗", "工资", "其他"]),
                },
                key="data_editor"
            )

            if st.button("💾 保存表格修改"):
                if not edited_df.equals(st.session_state.ledger_data):
                     if dm.save_data(edited_df, st.session_state.get('github_sha')):
                        st.session_state.ledger_data = edited_df
                        st.session_state.github_sha = dm.load_data()[1]
                        st.success("修改已同步至云端")
                else:
                    st.info("数据未发生变化")

            st.divider()
            
            c_chart1, c_chart2 = st.columns(2)
            
            df_chart = st.session_state.ledger_data.copy()
            df_chart['金额'] = pd.to_numeric(df_chart['金额'])
            df_exp = df_chart[df_chart['类型'] == '支出']
            
            with c_chart1:
                st.subheader("支出构成")
                if not df_exp.empty:
                    pie_data = df_exp.groupby("分类")['金额'].sum().reset_index()
                    st.bar_chart(pie_data, x="分类", y="金额", color="分类")
            
            with c_chart2:
                st.subheader("AI 洞察")
                if sf_api_key:
                    if st.button("🤖 生成月度分析报告"):
                        with st.spinner("AI 正在分析您的消费习惯..."):
                            summary_text = df_exp.to_csv(index=False)
                            client = get_llm_client(sf_api_key)
                            try:
                                res = client.chat.completions.create(
                                    model=TEXT_MODEL_NAME,
                                    messages=[
                                        {"role": "system", "content": "你是一个严厉但幽默的理财顾问。根据用户的支出数据，简短点评其消费习惯，并给出3条省钱建议。"},
                                        {"role": "user", "content": f"我的支出数据:\n{summary_text}"}
                                    ],
                                    max_tokens=4096
                                )
                                st.markdown(res.choices[0].message.content)
                            except Exception as e:
                                st.error(str(e))
                else:
                    st.caption("配置 API Key 后解锁 AI 分析")

if __name__ == "__main__":
    main()
