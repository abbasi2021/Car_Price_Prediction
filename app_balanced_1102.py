import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os

# Define the MLAPP class
class MLAPP:
    def __init__(self, model_path, scaler_path_X,scaler_path_y, ratio_excel_path):
        self.model_path = model_path
        # self.scaler_price_path = scaler_price_path
        self.scaler_path_X = scaler_path_X
        self.scaler_path_y = scaler_path_y
        self.ratio_excel_path = ratio_excel_path

        # Load model and scalers
        self.model = self._load_pickle(self.model_path)
        # self.scaler_price = self._load_pickle(self.scaler_price_path)
        self.scaler_X = self._load_pickle(self.scaler_path_X)
        self.scaler_y = self._load_pickle(self.scaler_path_y)

        self.ratio_mapping = self._load_ratio_mapping()

    def load_xgboost_model(self, path):
        try:
            model = xgb.XGBRegressor()
            model.load_model(path)
            return model
        except Exception as e:
            raise ValueError(f"Could not load XGBoost model: {e}")

    def _load_pickle(self, path):
        
            return pd.read_pickle( path)

    def _load_ratio_mapping(self):
        ratio_data = pd.read_excel(self.ratio_excel_path)
        ratio_data['type'] = ratio_data['type'].fillna('unknown')
        return ratio_data.set_index(['brand', 'model', 'type'])['ratio_type'].to_dict()

    def preprocess(self, data):
        """
        Preprocess the input data.

        Parameters:
        - data: pd.DataFrame, input data to preprocess.

        Returns:
        - preprocessed_data: pd.DataFrame, the data ready for prediction.
        """

        # Define column types
        
        numeric_columns = ['operation', 'product_year']
        categorical_columns = ['brand', 'model', 'color', 'motor_status','chassis_status', 'body_status', 'gearbox', 'fuel_type']

        # Convert columns to appropriate types
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int64')

        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category')

            
                # Before filling missing values, ensure 'unknown' is included in the categories
        if 'type' in data.columns:
            if not isinstance(data['type'].dtype, pd.CategoricalDtype):
                data['type'] = data['type'].astype('category')  # Ensure 'type' is categorical
            
            # Add 'unknown' to the categories if not already present
            if 'unknown' not in data['type'].cat.categories:
                data['type'] = data['type'].cat.add_categories('unknown')
            
            # Now you can safely fill missing values
            data['type'] = data['type'].fillna('unknown')




        
        # # Fill missing `type` values in the input data with 'unkn#############own'
        # data['type'] = data['type'].fillna('unknown')
        # data["type"] = data["type"].astype('category')
        
    
    
    

        # # Add sine and cosine transformations for 'month'
        # data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
        # data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)
        

        # Assign `ratio_type` using the mapping
        data['ratio_type'] = data.apply(
            lambda row: self.ratio_mapping.get((row['brand'], row['model'], row['type']), np.nan),
            axis=1
        )
        # Mappings for categorical fields
        mappings = {
            'motor_status': {"سالم": 3, "نیاز به تعمیر": 2, "تعویض شده": 1},
            'chassis_status': {"سالم و پلمپ": 3, "ضربه‌خورده": 2, "رنگ‌شده": 1},
            'body_status': {
                "سالم و بی‌خط و خش": 10, "خط و خش جزیی": 9, "صافکاری بی‌رنگ": 8,
                "رنگ‌شدگی، در ۱ ناحیه": 7, "رنگ‌شدگی، در ۲ ناحیه": 6, "رنگ‌شدگی": 5,
                "دوررنگ": 4, "تمام‌رنگ": 3, "تصادفی": 2, "اوراقی": 1
            },
            'gearbox': {"دنده‌ای": 0, "اتوماتیک": 1},
            # 'manufacturer':{"سایپا":0, "ایرانخودرو": 1},
            'brand': {
                "پژو": "peugeot", "پراید": "pride", "سمند": "samand", "کوییک": "quik",
                "تیبا": "tiba", "دنا": "dena", "ساینا": "saina", "رنو": "renault",
                "رانا": "runna", "شاهین": "shahin", "تارا": "tara"
            },
            
            'fuel_type': { "دوگانه‌سوز شرکتی": "cng_company", "دوگانه‌سوز دستی": "cng_manual","بنزینی": "oil"},

            'color': {
                "سفید": "white", "سفید صدفی": "white", "نقره‌ای": "silver",
                "مشکی": "black", "خاکستری": "grey", "نوک‌مدادی": "grey", "سایر": "other"
            }
        }


        # Map categorical fields
        for column, mapping in mappings.items():
            if column == 'color':
                data['color'] = data['color'].apply(lambda x: mapping.get(x, 'other'))  # Ensure 'other' is used for unknown colors
            else:
                data[column] = data[column].map(mapping)  # Map other categorical fields


        # One-hot encoding for categorical features
        for column, prefix in [('brand', 'brand'), ('color', 'color'), ('fuel_type', 'fuel_type')]:
            one_hot = pd.get_dummies(data[column], prefix=prefix)
            data = pd.concat([data, one_hot], axis=1)



        model_columns = self.model["feature_names"]
        for col in model_columns:
            if col not in data.columns:
                data[col] = 0  # Add missing columns with a value of 0

        data=data[model_columns]
        # data.filter(items=model_columns)      
                
        # #         # Scale 'price' and 'operation'
        # # # if 'price' in data.columns:
        data= self.scaler_X.transform(data)
        
      
        
        return data
    

    def prediction(self, data):
        preprocessed_data = self.preprocess(data)

        # model_columns = self.model["feature_names"]

        # preprocessed_data = preprocessed_data[model_columns]
        model_load=self.model["model"]
        predicted_scaled = model_load.predict(preprocessed_data)

        predictions = self.scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

        return predictions





# Set page configuration
st.set_page_config(page_title="Car Price Estimation", layout="wide")

# Add custom CSS for RTL direction
st.markdown(
    """
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply RTL direction to the title and markdown text
st.markdown('<h1 class="rtl">تخمین قیمت خودرو</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="rtl">اطلاعات مربوط به خودرو را وارد کنید</h3>', unsafe_allow_html=True)


# # Set up Streamlit interface
# st.set_page_config(page_title="Car Price Estimation", layout="wide")

# st.title(" تخمین قیمت خودرو")
# st.markdown("###  اطلاعات مربوط به خودروی خود را وارد کنید")

# Define dropdown options



brands=['کوییک', 'ساینا', 'پژو', 'دنا', 'تیبا', 'پراید', 'رنو',
       'سمند', 'تارا', 'رانا', 'شاهین']

models=['دنده\u200cای', '206 SD', 'پلاس', 'صندوق\u200cدار', 'پارس',
       'تندر 90 (L90 لوگان)', 'LX', 'اتوماتیک', 'سورن پلاس', '132', '131',
       '2 (هاچبک)', '405', '206', 'v1 پلاس', '207i', 'سورن', 'GX', '151',
       'معمولی', 'G', '141', '111', 'GL دنده ای', 'پلاس پانوراما', 'RS',
       'GXR', 'EL', 'SE', 'X7', 'GXH', 'پارس تندر', 'تندر 90 پلاس',
       'G CVT', 'دنا پلاس 5 دنده توربو', 'وی 4 ال ایکس', 'S اتومات']

types=['unknown','R Plus', 'EX', 'V ۸', 'S', 'اتوماتیک', 'SX', 'LX TU ۵', 'E ۲',
       'تیپ ۲ دنده\u200cای', 'EF ۷ بنزینی', 'ساده',
       'دوگانه\u200cسوز', 'SE', 'تیپ ۱ دنده\u200cای', 'سال',
       'جی ال ایکس (GLX)', 'ELX', 'تیپ ۲', 'پانوراما دنده\u200cای',
       'موتور جدید XU ۷ P', 'EF ۷', 'SLX موتورTU ۵', 'معمولی', 'تیپ ۱',
       'تیپ ۵', '۶ دنده توربو', 'دنده\u200cای',
       'GLX - دوگانه\u200cسوز CNG', 'موتور جدید XU ۷ P (سفارشی)', 'R',
       'SL', 'R پلاس', 'فول پلاس', 'GLX بنزینی', 'SX دوگانه\u200cسوز',
       'بنزینی GLX - TU ۵', 'تیپ ۳', 'SLX دوگانه\u200cسوز',
       'دنده\u200cای TU ۳', 'پلاس', 'ELX توربو شارژ', 'V ۲۰',
       'اتوماتیک TU ۵', 'جی ال آی (GLi)', 'اتوماتیک TU ۵ P', 'ELX XUM',
       'G', 'اتوماتیک MC', 'تیپ ۳ پانوراما', 'ELX (سفارشی) TU ۵',
       'GL - دوگانه\u200cسوز CNG', 'SLX موتور ۱۸۰۰', 'LX',
       'EX دوگانه\u200cسوز', 'E ۱', 'GL بنزینی', 'جی ال (GL)',
       'پانوراما اتوماتیک', 'TL', 'فول', 'بنزینی', 'V ۱',
       'LX دوگانه\u200cسوز', 'پانوراما اتوماتیک TU ۵ P', 'LE',
       'تیپ ۲ توربو', 'V ۹', 'E ۰', 'V ۲', 'GLi - دوگانه\u200cسوز CNG',
       'CNG', 'E ۲ دوگانه\u200cسوز', 'DLXI', 'GLX موتور ۱۶۰۰',
       'دوگانه\u200cسوز GLX - TU ۵', 'GLi بنزینی', 'V ۱۰',
       'GLX - دوگانه\u200cسوز LPG', 'تیپ ۱ توربو', 'تیپ ۶',
       'استیشن (لوگان)', 'استیشن', 'V ۱۹', 'E ۱ دوگانه\u200cسوز', 'i',
       'GL - دوگانه\u200cسوز LPG', 'تیپ L', 'V ۶', 'LPG',
       'GLi - دوگانه\u200cسوز LPG', 'تیپ ۴']

# manufacturer_=["سایپا","ایرانخودرو"]
product_years=[1390,1391,1392,1393,1394,1395,1396,1397,1398,1399,1400,1401,1402,1403]
colors=['سفید', 'سفید صدفی', 'مشکی', 'دلفینی', 'نقره\u200cای', 'آبی',
       'نوک\u200cمدادی', 'خاکستری', 'زرشکی', 'گیلاسی', 'آلبالویی', 'قرمز',
       'زرد', 'بادمجانی', 'قهوه\u200cای', 'کربن\u200cبلک', 'سرمه\u200cای',
       'ذغالی', 'اطلسی', 'بژ', 'مسی', 'کرم', 'نقرآبی', 'طوسی', 'تیتانیوم',
       'خاکی', 'بنفش', 'عنابی', 'سبز', 'سربی', 'زیتونی', 'طلایی',
       'پوست\u200cپیازی', 'نارنجی', 'موکا', 'برنز', 'یشمی']

motor_status=['سالم', 'تعویض شده', 'نیاز به تعمیر']
chassis_status=['سالم و پلمپ', 'ضربه\u200cخورده', 'رنگ\u200cشده']
body_status=['سالم و بی\u200cخط و خش', 'رنگ\u200cشدگی', 'خط و خش جزیی',
       'دوررنگ', 'تمام\u200cرنگ', 'صافکاری بی\u200cرنگ', 'تصادفی',
       'رنگ\u200cشدگی، در ۲ ناحیه', 'رنگ\u200cشدگی، در ۱ ناحیه', 'اوراقی']


gearboxes=['دنده\u200cای', 'اتوماتیک']
fuel_types=['بنزینی', 'دوگانه\u200cسوز شرکتی', 'دوگانه\u200cسوز دستی']
# operation=
# insurance_deadline=list(range(1,13))




# User input form
with st.form("car_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        brand = st.selectbox("برند", brands, key="brand")
        product_year = st.selectbox("سال تولید", product_years, key="product_year")
        motor_status_choice = st.selectbox("وضعیت موتور", motor_status, key="motor_status")
        
        
    with col2:
        model = st.selectbox("مدل", models, key="model")
        fuel_type = st.selectbox("نوع سوخت", fuel_types, key="fuel_type")
        chassis_status_choice = st.selectbox("وضعیت شاسی", chassis_status, key="chassis_status")
        # insurance=st.number_input("مانده بیمه", value=0,max_value=12,min_value=0 ,placeholder="مدت زمان باقیمانده از بیمه را وارد کنید",key="insurance")
    with col3:
        car_type = st.selectbox("تیپ", types, key="car_type")
        gearbox = st.selectbox("گیربکس", gearboxes, key="gearbox")
        body_status_choice = st.selectbox("وضعیت بدنه", body_status, key="body_status")
        
    # month=st.number_input("ماه فروش خودرو", value=0,max_value=12,min_value=0 ,placeholder="ماه فروش را وارد کنید")
    # manufacturer= st.selectbox("کارخانه", manufacturer_, key="manufacturer")


    col4, col5= st.columns(2)
    with col4:
        operation = st.number_input("کیلومتر ", min_value=0,max_value=700000, step=1000, key="operation")
        # color = st.selectbox("رنگ", colors, key="color")
    
    with col5:
        color = st.selectbox("رنگ", colors, key="color")       
    
    submitted = st.form_submit_button("پیش‌بینی قیمت")





# model path
model_path = "model_on_balanced_data-1102.pkl"
scaler_path_X = "scaler_X_for_balanced_data-1102.pkl"
scaler_path_y = "scaler_y_for_balanced_data-1102.pkl"
ratio_excel_path = "ratio-type.xlsx"
# Initialize app
app = MLAPP(model_path, scaler_path_X, scaler_path_y, ratio_excel_path)




app = MLAPP(model_path, scaler_path_X,scaler_path_y, ratio_excel_path)





car_data = pd.DataFrame([{
    "brand": brand,
    "model": model,
    "type": car_type,
    "product_year": product_year,
    "color": color,
    "motor_status": motor_status_choice,
    "chassis_status": chassis_status_choice,
    "body_status": body_status_choice,
    "gearbox": gearbox,
    "fuel_type": fuel_type,
    "operation": operation,
    # "insurance_deadline":10,
    # "month": 10,
    # "manufacturer": manufacturer  # Include manufacturer
}])


if submitted:


    # pre_data = app.preprocess(car_data)
    # st.dataframe(pre_data)

    prediction = app.prediction(car_data)
    # st.dataframe(prediction)
    st.success(f"قیمت پیش‌بینی‌شده: {prediction[0]:,.0f} تومان")
