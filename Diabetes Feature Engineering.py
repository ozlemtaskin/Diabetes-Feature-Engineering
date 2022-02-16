import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#!pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv(r"C:\Users\hp\PycharmProjects\pythonProject\6. HAFTA\diabetes.csv")
    return data

df = load()

#Genel resmi inceleme

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


#Numerik ve kategorik değişkenleri yakalama

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Numerik ve kategorik değişken analizleri

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, plot=True)


#Hedef değişken analizi

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"Outcome",col)

df.groupby("Outcome").agg({"mean"})

#Aykırı gözlem analizi

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col_names in df:
    print(col_names, check_outlier(df, col_names))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col_names in df:
    print(col_names ,"\n", grab_outliers(df, col_names))


#Çok değişkenli aykırı değer analizi: local outlier factor

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[10]
df[df_scores < th]
df[df_scores < th].shape

#Eksik gözlem analizi

df.isnull().sum()

#Korelasyon analizi

# Adım 7: Korelasyon analizi yapınız.
corr = df[num_cols].corr()
corr

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)


#Görev 2 : Feature Engineering

#Eksik değerler için işlemler
df[df["Insulin"]==0].shape
df["Insulin"].replace(0,np.NaN,inplace=True)
df["Insulin"].describe().T

df[df["Glucose"]==0].shape
df["Glucose"].replace(0,np.NaN,inplace=True)
df["Glucose"].describe().T

df.groupby("Outcome").agg({"Glucose":"mean"})
df["Glucose"].isnull().sum()

df.loc[(df["Glucose"].isnull()) & (df["Outcome"]==0), "Glucose" ]= df.groupby("Outcome")["Glucose"].mean()[0]
df.loc[(df["Glucose"].isnull()) & (df["Outcome"]==1), "Glucose" ]= df.groupby("Outcome")["Glucose"].mean()[1]

df.groupby("Outcome").agg({"Insulin":"mean"})
df["Insulin"].isnull().sum()

df.loc[(df["Insulin"].isnull()) & (df["Outcome"]==0), "Insulin" ]= df.groupby("Outcome")["Insulin"].mean()[0]
df.loc[(df["Insulin"].isnull()) & (df["Outcome"]==1), "Insulin" ]= df.groupby("Outcome")["Insulin"].mean()[1]

#Aykırı değerler için işlemler

for col_names in df:
    print(col_names, check_outlier(df, col_names))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

#Yeni değişkenler

#age level
df.loc[(df["Age"]<30) , "New_Age_Cat"]= "young"
df.loc[(df["Age"]>=30) & (df["Age"]<50), "New_Age_Cat" ]= "mature"
df.loc[(df["Age"]>= 50), "New_Age_Cat"]= "senior"

#bmi level

#Underweight	Below 18.5
#Healthy weight	18.5–24.9
#Overweight	25.0–29.9
#Obese	30.0 and higher

df.loc[(df["BMI"]< 18.5), "New_BMI_Cat"] ="underweight"
df.loc[(df["BMI"]>=18.5) & (df["BMI"]<24.9), "New_BMI_Cat"]="healthy weight"
df.loc[(df["BMI"]>=24.9) & (df["BMI"]<29.9), "New_BMI_Cat"]="overweight"
df.loc[(df["BMI"]>=29.9) , "New_BMI_Cat"]="obese"

#glucose level

df["New_Glucose_Cat"]=pd.cut(x=df["Glucose"], bins=[0,140,200,300], labels=["normal", "prediabetes","diabetes"])

df.head()

#pregnancies level

df["Pregnancies"].mean()

df.loc[(df["Pregnancies"]== 0), "New_Pregnancies_Cat"] ="no pregnancy"
df.loc[(df["Pregnancies"]>0) & (df["Pregnancies"]<=4), "New_Pregnancies_Cat"]="average or less"
df.loc[(df["Pregnancies"]>4), "New_Pregnancies_Cat"]="more than average"


#BloodPressure level
df.loc[(df["BloodPressure"]<70), "New_BloodPress_Cat"] ="less than ideal"
df.loc[(df["BloodPressure"]>=70) & (df["BloodPressure"]<=90), "New_BloodPress_Cat"]="ideal"
df.loc[(df["BloodPressure"]>90), "New_BloodPress_Cat"]="more than ideal"

df.head(20)

#age x bmi
df.loc[(df["Age"]<30) & (df["BMI"]<18.5), "New_Age_Bmi_Cat"]= "underweight young"
df.loc[(df["Age"]>=30) & (df["Age"]<50) & (df["BMI"]<18.5), "New_Age_Bmi_Cat"]= "underweight mature"
df.loc[(df["Age"]>=50) & (df["BMI"]<18.5), "New_Age_Bmi_Cat"]= "underweight senior"
df.loc[(df["Age"]<30) & (df["BMI"]>=18.5) & (df["BMI"]<24.9) , "New_Age_Bmi_Cat"]= "healthy young"
df.loc[(df["Age"]>=30) & (df["Age"]<50) & (df["BMI"]>=18.5) & (df["BMI"]<24.9) , "New_Age_Bmi_Cat"]= "healthy mature"
df.loc[(df["Age"]>=50) & (df["BMI"]>=18.5) & (df["BMI"]<24.9) , "New_Age_Bmi_Cat"]= "healthy senior"
df.loc[(df["Age"]<30) & (df["BMI"]>=24.9) & (df["BMI"]<29.9) , "New_Age_Bmi_Cat"]= "overweight young"
df.loc[(df["Age"]>=30) & (df["Age"]<50) & (df["BMI"]>=24.9) & (df["BMI"]<29.9) , "New_Age_Bmi_Cat"]= "overweight mature"
df.loc[(df["Age"]>=50) & (df["BMI"]>=24.9) & (df["BMI"]<29.9) , "New_Age_Bmi_Cat"]= "overweight senior"
df.loc[(df["Age"]<30) & (df["BMI"]>=29.9) , "New_Age_Bmi_Cat"]= "obese young"
df.loc[(df["Age"]>=30) & (df["Age"]<50) & (df["BMI"]>=29.9) , "New_Age_Bmi_Cat"]= "obese mature"
df.loc[(df["Age"]>=50) & (df["BMI"]>=29.9) , "New_Age_Bmi_Cat"]= "obese senior"


# insulin level
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Anormal"

df["New_Insulin_Score"] = df.apply(set_insulin, axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


df.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Encoding İşlemleri

#Label Encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#One Hot Encoder

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

#Standartlaştırma

#Standard Scaler


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
#Model

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#Önemli değişkenleri tespit etme

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train, num=30)