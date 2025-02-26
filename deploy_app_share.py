#######我们将加载训练好的模型并在 Streamlit 中进行部署
import streamlit as st
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型和标准化器
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载模型
model_path = os.path.join(current_dir, 'copd_xgb_model_v2.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到：{model_path}")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"加载模型失败: {str(e)}")
# 加载标准化器（需确保保存了scaler）
scaler_path = os.path.join(current_dir, 'standard_scaler.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# 配置页面
st.set_page_config(page_title="COPD预测系统", layout="wide")
st.title("COPD风险预测系统")
st.markdown("""
**说明**：本系统通过呼吸模式特征预测慢性阻塞性肺疾病（COPD）风险  
输入患者特征后，点击预测按钮获取结果
""")

# 侧边栏输入
with st.sidebar:
    st.header("患者基本信息")
    # 基本信息
    age = st.number_input("年龄 (age)", min_value=18, max_value=89, value=50,
                          help="患者年龄，范围：18-89岁")
    BMI = st.number_input("BMI (kg/m²)", min_value=13.6, max_value=34.8, value=22.0,
                          step=0.1, format="%.1f", help="体质指数，范围：13.6-34.8 kg/m²")
    gender = st.radio("性别", options=[("女性", 0), ("男性", 1)],
                      format_func=lambda x: x[0])[1]
    smoke = st.radio("吸烟史", options=[("无", 0), ("有", 1)],
                     format_func=lambda x: x[0])[1]

    # 坐姿呼吸特征
    st.header("静坐呼吸特征（Sitting Features）")
    with st.expander("展开坐姿参数"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("基本参数")
            f1 = st.slider("呼吸频率 (f1)", 9.0, 46.4, 20.0,
                           help="静坐平均呼吸频率 9.000-46.381次/分")
            A1 = st.slider("呼吸幅度 (A1)", 0.217, 18.070, 5.0,
                           help="静坐平均呼吸幅度 0.217-18.070")
            r1 = st.slider("吸呼比 (r1)", 0.323, 0.869, 0.6,
                           help="静坐平均吸呼比 0.323-0.869")
            tt1 = st.slider("呼气时间比 (tt1)", 0.038, 0.345, 0.15,
                            help="静坐呼气峰流量时间比 0.038-0.345")
            eta1 = st.slider("呼气峰幅比 (eta1)", 0.144, 0.403, 0.25,
                             help="静坐呼气流量峰值幅度比 0.144-0.403")
            area1 = st.slider("峰面积比 (area1)", 0.003, 0.081, 0.02,
                              format="%.3f", help="静坐呼气峰面积比 0.003-0.081")

        with col2:
            st.subheader("非线性指数")
            nl11 = st.slider("第一模态 (nl11)", 0.007, 0.330, 0.1, step=0.001,
                             format="%.3f", help="第一模态非线性指数 0.007-0.330")
            nl12 = st.slider("第二模态 (nl12)", 0.016, 0.210, 0.1, step=0.001,
                             format="%.3f", help="第二模态非线性指数 0.016-0.210")
            nl13 = st.slider("第三模态 (nl13)", 0.008, 0.615, 0.1, step=0.001,
                             format="%.3f", help="第三模态非线性指数 0.008-0.615")

        with col3:
            st.subheader("主频参数")
            omega_mean11 = st.slider("第一主频 (omega_mean11)", 0.111, 0.572, 0.3,
                                     step=0.001, format="%.3f", help="第一模态主频 0.111-0.572")
            omega_mean12 = st.slider("第二主频 (omega_mean12)", 0.055, 0.202, 0.1,
                                     step=0.001, format="%.3f", help="第二模态主频 0.055-0.202")
            omega_mean13 = st.slider("第三主频 (omega_mean13)", 0.018, 0.086, 0.05,
                                     step=0.001, format="%.3f", help="第三模态主频 0.018-0.086")

    # 站立呼吸特征
    st.header("静站呼吸特征（Standing Features）")
    with st.expander("展开站姿参数"):
        col4, col5, col6 = st.columns(3)

        with col4:
            st.subheader("基本参数")
            f2 = st.slider("呼吸频率 (f2)", 9.628, 42.445, 20.0,
                           help="静站平均呼吸频率 9.628-42.445次/分")
            A2 = st.slider("呼吸幅度 (A2)", 0.755, 19.493, 5.0,
                           help="静站平均呼吸幅度 0.755-19.493")
            r2 = st.slider("吸呼比 (r2)", 0.378, 0.911, 0.6,
                           help="静站平均吸呼比 0.378-0.911")
            tt2 = st.slider("呼气时间比 (tt2)", 0.051, 0.317, 0.15,
                            help="静站呼气峰流量时间比 0.051-0.317")
            eta2 = st.slider("呼气峰幅比 (eta2)", 0.157, 0.450, 0.25,
                             help="静站呼气流量峰值幅度比 0.157-0.450")
            area2 = st.slider("峰面积比 (area2)", 0.003, 0.071, 0.02,
                              format="%.3f", help="静站呼气峰面积比 0.003-0.071")

        with col5:
            st.subheader("非线性指数")
            nl21 = st.slider("第一模态 (nl21)", 0.006, 0.348, 0.1, step=0.001,
                             format="%.3f", help="第一模态非线性指数 0.006-0.348")
            nl22 = st.slider("第二模态 (nl22)", 0.012, 0.211, 0.1, step=0.001,
                             format="%.3f", help="第二模态非线性指数 0.012-0.211")
            nl23 = st.slider("第三模态 (nl23)", 0.014, 0.340, 0.1, step=0.001,
                             format="%.3f", help="第三模态非线性指数 0.014-0.340")

        with col6:
            st.subheader("主频参数")
            omega_mean21 = st.slider("第一主频 (omega_mean21)", 0.096, 0.578, 0.3,
                                     step=0.001, format="%.3f", help="第一模态主频 0.096-0.578")
            omega_mean22 = st.slider("第二主频 (omega_mean22)", 0.053, 0.193, 0.1,
                                     step=0.001, format="%.3f", help="第二模态主频 0.053-0.193")
            omega_mean23 = st.slider("第三主频 (omega_mean23)", 0.017, 0.085, 0.05,
                                     step=0.001, format="%.3f", help="第三模态主频 0.017-0.085")

# 构建特征数据框
input_data = pd.DataFrame([{
    # 基本信息
    'age': age,
    'BMI': BMI,
    'gender': gender,
    'smoke': smoke,

    # 坐姿特征
    'f1': f1, 'A1': A1, 'r1': r1, 'tt1': tt1, 'eta1': eta1, 'area1': area1,
    'nl11': nl11, 'nl12': nl12, 'nl13': nl13,
    'omega_mean11': omega_mean11, 'omega_mean12': omega_mean12, 'omega_mean13': omega_mean13,

    # 站姿特征
    'f2': f2, 'A2': A2, 'r2': r2, 'tt2': tt2, 'eta2': eta2, 'area2': area2,
    'nl21': nl21, 'nl22': nl22, 'nl23': nl23,
    'omega_mean21': omega_mean21, 'omega_mean22': omega_mean22, 'omega_mean23': omega_mean23
}], columns=model.feature_names_in_)

# 预处理
cont_features = [f for f in input_data.columns if f not in ['gender', 'smoke']]
input_data[cont_features] = scaler.transform(input_data[cont_features])

# 预测和展示
if st.button("开始预测"):
    # 执行预测
    proba = model.predict_proba(input_data)[0][1]
    prediction = "高危 (COPD)" if proba >= 0.5 else "低危 (非COPD)"

    # 主界面展示
    st.subheader("预测结果")
    col5, col6 = st.columns(2)
    with col5:
        st.metric("风险评估", f"{prediction}", f"{proba * 100:.1f}% 概率")
    with col6:
        st.write("""
        **结果解读**：  
        - 高风险 (≥50%概率): 建议进行肺功能检查  
        - 低风险 (<50%概率): 建议保持健康监测
        """)

    # SHAP解释
    st.subheader("特征影响分析")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    with st.expander("查看详细解释"):
        try:
            st.write("### 特征贡献力图")
            plt.close('all')

            # 生成力图
            fig = shap.plots.force(
                shap_values[0],
                feature_names=model.feature_names_in_,
                matplotlib=True,
                show=False,
                figsize=(20, 6)
            )

            # ==== 数值格式化修改 ====
            ax = fig.gca()
            for text in ax.texts:
                content = text.get_text()
                if '=' in content:
                    try:
                        # 拆分特征名和数值
                        parts = content.split('=')
                        feature_name = '='.join(parts[:-1])
                        raw_value = parts[-1].strip()

                        # 格式化数值
                        formatted_value = f"{float(raw_value):.2f}"
                        new_text = f"{feature_name}= {formatted_value}"
                        text.set_text(new_text)
                    except:
                        pass
            # ==== 修改结束 ====

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"特征力图生成失败: {str(e)}")

        # 特征重要性
        st.write("### 特征重要性排序")
        shap.plots.bar(shap_values, max_display=15, show=False)
        st.pyplot(bbox_inches='tight')
        plt.clf()

# 注意事项
st.markdown("---")
st.warning("""
**临床使用注意事项**：  
1. 本预测结果仅供参考，不能替代专业医疗诊断  
2. 高风险患者建议结合肺功能检查确认  
3. 系统预测准确率：验证集AUC=0.908，灵敏度=83.9%，特异度=81.9%
""")

