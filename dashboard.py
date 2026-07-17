
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------
# CONFIG TRANG DASHBOARD
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="GP Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 DASHBOARD PHÂN TÍCH KINH TẾ LƯỢNG - TRƯỜNG THPT GP")
st.markdown("---")

# ---------------------------------------------------------------------
# ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("data.xlsx")
    df['ln_absences'] = np.log(df['absences'] + 1)
    df['ln_absences_squared'] = df['ln_absences']**2
    return df

try:
    df = load_data()
except:
    st.error("Không tìm thấy file dữ liỆu!")
    st.stop()

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Bộ Lọc")
category_filter = st.sidebar.selectbox("Lọc theo Khu vực sống (Address):", ["Tất cả", "Thành thị (Urban)", "Nông thôn (Rural)"])

if category_filter == "Thành thị (Urban)":
    filtered_df = df[df['address'] == 1]
elif category_filter == "Nông thôn (Rural)":
    filtered_df = df[df['address'] == 0]
else:
    filtered_df = df

# ---------------------------------------------------------------------
# MỤC 1: TỔNG QUAN MÔ HÌNH (M4)
# ---------------------------------------------------------------------
st.header("📊 1. Tổng quan mô hình tối ưu (M4: Robust SE)")

# Công thức mô hình M4 đã xác định tại Yéu cầu 11
formula = 'avgscore ~ age + failures + studytime_3 + sex + schoolsup + famsup + higher + ln_absences + ln_absences_squared + failures:schoolsup + age:failures'
model = smf.ols(formula, data=filtered_df).fit(cov_type='HC3')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Hệ số xác định R²", value=f"{model.rsquared:.4f}")
with col2:
    st.metric(label="R² hiệu chỉnh", value=f"{model.rsquared_adj:.4f}")
with col3:
    st.metric(label="Chỉ số AIC", value=f"{model.aic:.2f}")
with col4:
    st.metric(label="Mẫu quan sát (N)", value=f"{len(filtered_df)}")

# ---------------------------------------------------------------------
# MỤC 2: TÁC ĐỘNG BIÊÊ
# ---------------------------------------------------------------------
st.header("2. Mức độ ảnh hưởng của các nhân tố")
coef_df = pd.DataFrame({
    'Nhân tố': model.params.index[1:],
    'Beta': model.params.values[1:],
    'p-value': model.pvalues.values[1:]
})
coef_df['Tác đỘng'] = np.where(coef_df['Beta'] >= 0, 'Tích cực (+)', 'Tiéu cực (-)')

fig = px.bar(coef_df, x='Beta', y='Nhân tố', color='Tác đỘng', orientation='h',
             title="Hệ số hồi quy từ mô hình M4")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# MỤC 4: DỰ BÁO
# ---------------------------------------------------------------------
st.header("4. Mô phỏng dự báo điểm số")
c1, c2, c3 = st.columns(3)
with c1:
    in_age = st.number_input("Tuổi (age):", 15, 22, 17)
    in_fail = st.slider("Số lần trượt (failures):", 0, 3, 0)
    in_sex = st.selectbox("Giớ tính:", ["Nữ (0)", "Nam (1)"])
with c2:
    in_abs = st.number_input("Số buổi vắng (absences):", 0, 50, 5)
    in_sup = st.selectbox("Hỗ trợ trường (schoolsup):", ["Không (0)", "Có (1)"])
    in_famsup = st.selectbox("Hỗ trợ gia đình (famsup):", ["Không (0)", "Có (1)"])
with c3:
    in_study = st.selectbox("Thối gian học hàng tuần:", ["Dưới 5h", "5-10h (Mức 3)", "Trên 10h"])
    in_higher = st.checkbox("Muốn học cao hơn (higher)", True)

# Chuyển đổi input
sex_v = 1 if "Nam" in in_sex else 0
sup_v = 1 if "Có" in in_sup else 0
fam_v = 1 if "Có" in in_famsup else 0
high_v = 1 if in_higher else 0
st3_v = 1 if "5-10h" in in_study else 0
ln_abs = np.log(in_abs + 1)

# Tạo DataFrame cho dự báo để statsmodels tự tính ln^2 và interaction
input_data = pd.DataFrame({
    'age': [in_age], 'failures': [in_fail], 'studytime_3': [st3_v],
    'sex': [sex_v], 'schoolsup': [sup_v], 'famsup': [fam_v], 'higher': [high_v],
    'ln_absences': [ln_abs], 'ln_absences_squared': [ln_abs**2]
})

pred_score = model.predict(input_data)[0]
pred_score = max(0.0, min(20.0, pred_score))

st.markdown(f"""
<div style='background-color:#f0f3f4; padding:20px; border-radius:10px; text-align:center;'>
    <h2 style='color:#2c3e50;'> ĐIỂM DỰ BÁO: {pred_score:.2f} / 20</h2>
</div>
""", unsafe_allow_html=True)
