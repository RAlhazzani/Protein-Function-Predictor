import streamlit as st
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# الجزء 1: استخراج الخصائص الحيوية (Bioinformatics Part)
# هذا الكود يحول "الحروف" إلى "أرقام" يفهمها الكمبيوتر
# ---------------------------------------------------------
def get_protein_features(sequence):
    """
    دالة تأخذ تسلسل البروتين وتعيد خصائصه الكيميائية والفيزيائية
    """
    try:
        # التأكد من أن التسلسل حروف كبيرة (Upper case)
        seq = sequence.upper()
        
        # استخدام مكتبة BioPython لتحليل البروتين
        analysed_seq = ProteinAnalysis(seq)
        
        # استخراج الخصائص (هذه هي الـ Features للموديل)
        features = {
            'Molecular_Weight': analysed_seq.molecular_weight(), # الوزن الجزيئي
            'Aromaticity': analysed_seq.aromaticity(),           # العطرية
            'Instability_Index': analysed_seq.instability_index(), # معامل عدم الاستقرار
            'Isoelectric_Point': analysed_seq.isoelectric_point(), # نقطة التعادل الكهربائي
            'Helix_Fraction': analysed_seq.secondary_structure_fraction()[0] # نسبة الحلزون
        }
        return features
    except Exception as e:
        return None

# ---------------------------------------------------------
# الجزء 2: تجهيز البيانات وتدريب الموديل (Machine Learning Part)
# في الوضع الحقيقي، نستبدل هذا الجزء بقراءة ملف CSV
# ---------------------------------------------------------
@st.cache_resource  # هذا السطر يجعل Streamlit يحفظ الموديل في الذاكرة لسرعة الأداء
def train_model():
    # 1. إنشاء بيانات وهمية للتدريب (لغرض التجربة الفورية)
    # سنفترض أننا نصنف البروتينات إلى: (Soluble) و (Insoluble)
    data = []
    labels = []
    
    # توليد 100 بروتين عشوائي (محاكاة)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(100):
        # إنشاء تسلسل عشوائي
        seq_len = np.random.randint(50, 200)
        seq = "".join(np.random.choice(list(amino_acids), seq_len))
        
        # استخراج خصائصه
        feats = get_protein_features(seq)
        
        # وضع قاعدة تصنيف "وهمية" بسيطة لكي يتعلم الموديل شيئاً:
        # إذا الوزن الجزيئي عالي والـ Isoelectric point عالي -> نعتبره Insoluble (1)
        # وإلا -> Soluble (0)
        # (ملاحظة: هذا تبسيط علمي شديد فقط لغرض الكود)
        if feats['Molecular_Weight'] > 100 and feats['Isoelectric_Point'] > 6:
            label = "Insoluble (غير ذائب)"
        else:
            label = "Soluble (ذائب)"
            
        data.append(list(feats.values()))
        labels.append(label)

    # 2. تحويل البيانات لجدول (DataFrame)
    df = pd.DataFrame(data, columns=['Molecular_Weight', 'Aromaticity', 'Instability_Index', 'Isoelectric_Point', 'Helix_Fraction'])
    
    # 3. تدريب الموديل
    X = df  # الخصائص (Inputs)
    y = labels  # النتيجة المطلوبة (Outputs)
    
    # تقسيم البيانات لتدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # إنشاء الموديل (Random Forest)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # حساب الدقة
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, acc

# ---------------------------------------------------------
# الجزء 3: واجهة المستخدم (Web App Interface)
# هذا ما سيظهر على الشاشة
# ---------------------------------------------------------

# إعداد الصفحة
st.set_page_config(page_title="Protein Classifier AI", page_icon="🧬")

# العنوان والمقدمة
st.title("🧬 AI Protein Function Predictor")
st.markdown("""
هذا التطبيق يستخدم **الذكاء الاصطناعي (Machine Learning)** لتوقع خصائص البروتين 
بناءً على خصائصه الفيزيائية والكيميائية.
""")

# الشريط الجانبي (Sidebar)
st.sidebar.header("معلومات الموديل")
model, accuracy = train_model()
st.sidebar.success(f"دقة الموديل الحالي: {accuracy * 100:.1f}%")
st.sidebar.info("تم التدريب باستخدام خوارزمية Random Forest")

# منطقة إدخال البيانات
input_sequence = st.text_area("أدخل تسلسل البروتين (Sequence) هنا:", height=150, placeholder="Example: MVLSPADKTNVKAAWGKVGAHAGEYGAE...")

# زر التوقع
if st.button("تحليل وتوقع النتيجة 🚀"):
    if input_sequence:
        # 1. استخراج الخصائص
        features = get_protein_features(input_sequence)
        
        if features:
            # عرض الخصائص المستخرجة
            st.subheader("1. الخصائص المستخرجة (Bio-Features):")
            features_df = pd.DataFrame([features])
            st.table(features_df)
            
            # 2. التوقع باستخدام الموديل
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df).max()
            
            # عرض النتيجة
            st.subheader("2. نتيجة الذكاء الاصطناعي:")
            
            # تلوين النتيجة
            if "Soluble" in prediction:
                st.success(f"النتيجة المتوقعة: **{prediction}**")
            else:
                st.warning(f"النتيجة المتوقعة: **{prediction}**")
                
            st.write(f"نسبة الثقة (Confidence): **{probability*100:.2f}%**")
            
        else:
            st.error("عذراً، التسلسل المدخل يحتوي على رموز غير صحيحة. تأكد من استخدام الأحماض الأمينية فقط.")
    else:
        st.warning("الرجاء إدخال تسلسل بروتين أولاً.")

# تذييل الصفحة
st.markdown("---")
st.caption("Developed by Raneem Alhazzani | 2600200@uj.edu.sa")