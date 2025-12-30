import streamlit as st
import os
import sys
import subprocess
import time

# --- 核心修复：启动时自动创建必要目录 ---
# Git 不会上传空文件夹，导致云端运行时因找不到目录报错 (RuntimeError: Directory does not exist)。
# 这段代码会在应用启动瞬间自动创建它们，无需你在仓库里手动操作。
REQUIRED_DIRS = ['temp', 'static', 'assets']
for dir_name in REQUIRED_DIRS:
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print(f"已自动创建目录: {dir_name}") # 打印日志方便调试
        except Exception as e:
            print(f"创建目录 {dir_name} 失败: {e}")

# --- 依赖库检查 ---
# 既然 requirements.txt 已包含 pymupdf，这里作为最后的“保底”措施
try:
    import fitz  # PyMuPDF
except ImportError:
    st.warning("检测到 PyMuPDF 未安装，正在尝试自动修复...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
        import fitz
        st.success("PyMuPDF 已自动安装！请刷新页面。")
    except Exception as e:
        st.error(f"无法安装 PyMuPDF。请确保 requirements.txt 中包含 'pymupdf'。\n错误: {e}")
        st.stop()

# --- 应用主逻辑 ---

st.set_page_config(page_title="财务文档分析器", layout="wide")

st.title("💰 财务文档分析工具")

# 说明区域
with st.expander("ℹ️ 关于此应用", expanded=False):
    st.write("此应用用于解析财务 PDF 报表。如果遇到目录错误，系统已尝试自动修复。")

uploaded_file = st.file_uploader("请上传财务报表 (PDF)", type=["pdf"])

if uploaded_file:
    # 确保文件名安全，防止路径问题
    safe_filename = "".join([c for c in uploaded_file.name if c.isalpha() or c.isdigit() or c in (' ', '.', '_')]).strip()
    temp_path = os.path.join("temp", safe_filename)
    
    # 保存文件
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ 文件已上传: {uploaded_file.name}")
        
        # 开始解析
        try:
            doc = fitz.open(temp_path)
            
            # 布局：左侧信息，右侧预览
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.info("📄 文档概览")
                st.write(f"**总页数:** {doc.page_count}")
                st.markdown("**元数据:**")
                st.json(doc.metadata)

            with col2:
                st.subheader("👀 内容预览 (第1页)")
                if doc.page_count > 0:
                    page = doc.load_page(0)
                    
                    # 文本预览
                    text = page.get_text()
                    st.text_area("提取的文本内容", text, height=300)
                    
                    # 图片预览
                    st.markdown("**页面截图:**")
                    pix = page.get_pixmap()
                    st.image(pix.tobytes(), caption=f"第 1 页 / 共 {doc.page_count} 页", use_container_width=True)
            
            doc.close()
            
        except Exception as e:
            st.error(f"❌ 解析 PDF 时发生错误: {e}")
            
    except Exception as e:
        st.error(f"❌ 保存文件失败: {e}")
        
    finally:
        # 清理逻辑：处理完后删除临时文件，保持环境整洁
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass # 如果删除失败也没关系，那是系统临时文件

else:
    # 空状态提示
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 50px;">
        <h3>👋 欢迎使用</h3>
        <p>请在上方上传 PDF 文件开始分析</p>
    </div>
    """, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.caption("Environment: Streamlit Cloud | Engine: PyMuPDF")
